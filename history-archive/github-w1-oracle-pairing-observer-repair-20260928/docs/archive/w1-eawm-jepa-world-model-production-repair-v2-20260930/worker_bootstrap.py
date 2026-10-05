"""Linux child entrypoint: no orphan execution after supervisor death."""
import ctypes,os,signal,sys

def main():
    expected=int(sys.argv[1]);target=sys.argv[2]
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG failed')
    if os.getppid()!=expected:raise RuntimeError('Supervisor died before child guard initialization')
    os.execv(sys.executable,[sys.executable,'-B',target])
if __name__=='__main__':main()
