# Registration Start Handshake Test Evidence

Scope: synthetic JSON and bottom-channel tests only. No model, simulator, SSH, network, server registration, or production runner was invoked.

Command:

```powershell
python -m unittest -v test_registration_start_handshake
```

Observed output:

```text
test_controller_receipt_timeout_and_eof_are_rejected (test_registration_start_handshake.RegistrationStartHandshakeTests.test_controller_receipt_timeout_and_eof_are_rejected) ... ok
test_controller_rejects_mismatched_receipt_without_reporting (test_registration_start_handshake.RegistrationStartHandshakeTests.test_controller_rejects_mismatched_receipt_without_reporting) ... ok
test_controller_reports_receipt_but_sends_no_ack_on_refusal (test_registration_start_handshake.RegistrationStartHandshakeTests.test_controller_reports_receipt_but_sends_no_ack_on_refusal) ... ok
test_receipt_contains_only_registered_identity_and_binding_digests (test_registration_start_handshake.RegistrationStartHandshakeTests.test_receipt_contains_only_registered_identity_and_binding_digests) ... ok
test_registered_report_ack_running_order_over_channel (test_registration_start_handshake.RegistrationStartHandshakeTests.test_registered_report_ack_running_order_over_channel) ... ok
test_server_rejects_bad_binding_eof_and_timeout_before_running (test_registration_start_handshake.RegistrationStartHandshakeTests.test_server_rejects_bad_binding_eof_and_timeout_before_running) ... ok
test_server_rejects_registration_binding_before_receipt (test_registration_start_handshake.RegistrationStartHandshakeTests.test_server_rejects_registration_binding_before_receipt) ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.023s

OK
exit_code=0
```

After this run, the server helper was amended to expose the already-registered job on handshake exceptions and to accept an `on_registered` state-capture callback. Per parent instruction, the tests were not rerun after that small API addition; the captured output therefore verifies the preceding seven-test implementation, not the final amended bytes.
