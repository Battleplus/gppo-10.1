import threading
import time
import unittest
from transport_wall_budget import WallBudget, WallDeadlineExceeded
from acceptance_entry import _remaining_wall_request, AcceptanceError


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now


class WallBudgetTests(unittest.TestCase):
    def test_preflight_upload_bootstrap_and_download_share_one_deadline(self):
        clock=Clock()
        budget=WallBudget(180,closing_reserve=5,clock=clock)
        for stage,duration in [('preflight',80),('upload',20),('bootstrap',60)]:
            with budget.phase(stage):clock.now+=duration
        self.assertEqual(budget.remaining(),15)
        with self.assertRaises(WallDeadlineExceeded):
            with budget.phase('download'):clock.now+=16
        self.assertEqual(budget.phases[-1]['outcome'],'error')

    def test_socket_timeout_is_capped_by_global_remainder(self):
        clock=Clock()
        budget=WallBudget(180,closing_reserve=5,clock=clock)
        clock.now=160
        self.assertEqual(budget.timeout(180),15)
        self.assertEqual(budget.timeout(10),10)

    def test_no_time_means_no_remote_operation(self):
        clock=Clock()
        budget=WallBudget(180,closing_reserve=5,clock=clock)
        clock.now=176
        with self.assertRaises(WallDeadlineExceeded):budget.timeout(30)

    def test_remote_receives_smaller_cap_and_keeps_closing_reserve(self):
        clock=Clock()
        budget=WallBudget(180,closing_reserve=5,clock=clock)
        clock.now=100
        self.assertEqual(budget.remote_allowance(150,remote_closing_reserve=5),70)
        clock.now=170
        with self.assertRaises(WallDeadlineExceeded):
            budget.remote_allowance(150,remote_closing_reserve=5)

    def test_real_timer_unblocks_stalled_owned_transport_only(self):
        event=threading.Event()
        class Channel:
            def settimeout(self,seconds):self.timeout=seconds
        class SFTP:
            def get_channel(self):return Channel()
            def stat(self,path):
                if not event.wait(1):raise AssertionError('timer did not interrupt')
                raise OSError('owned transport closed')
            def close(self):pass
        class Client:
            close_count=0
            def open_sftp(self):return SFTP()
            def close(self):self.close_count+=1;event.set()
        client=Client()
        budget=WallBudget(.15,closing_reserve=.05)
        wrapped=budget.bind(client)
        started=time.monotonic()
        try:
            with self.assertRaises(OSError):
                with wrapped.open_sftp() as sftp:sftp.stat('/fixture')
            self.assertTrue(budget.expired.is_set())
            self.assertEqual(client.close_count,1)
            self.assertLess(time.monotonic()-started,.8)
        finally:budget.close()

    def test_finished_operation_cancels_timer(self):
        class Client:
            closed=False
            def close(self):self.closed=True
        budget=WallBudget(180,closing_reserve=5)
        client=Client()
        budget.bind(client)
        budget.close()
        self.assertFalse(client.closed)
        self.assertFalse(budget.expired.is_set())

    def test_remote_cap_changes_only_runtime_copy(self):
        request={'limits':{'server_wall_seconds':150,'terminalization_reserve_wall_seconds':5,'calls':{'updates':18}}}
        revised=_remaining_wall_request(request,70)
        self.assertEqual(revised['limits']['server_wall_seconds'],70)
        self.assertEqual(request['limits']['server_wall_seconds'],150)
        self.assertEqual(revised['limits']['calls'],request['limits']['calls'])

    def test_remote_cap_rejects_expansion_nan_and_no_closing_time(self):
        request={'limits':{'server_wall_seconds':150,'terminalization_reserve_wall_seconds':5}}
        for value in (151,float('nan'),float('inf'),5,0,-1):
            with self.subTest(value=value),self.assertRaises(AcceptanceError):
                _remaining_wall_request(request,value)

if __name__=='__main__':unittest.main()
