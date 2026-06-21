# Dimension SEC — Security & input validation

numbox is a library (not a network service), so "security" here means **memory-safety-relevant input
validation and unsafe-construct hygiene at the C boundary**: places where bad or adversarial input —
or a malformed buffer, format string, path, or SQL — leads to corruption, a crash, disclosure, or
unintended execution. Applied to C-binding targets (sqlite + libc + abi/call + lib-loading + C-string
utils).

## What to hunt

- **Bounds & length validation.** Buffer reads/writes whose length comes from caller-controlled or
  C-returned data without a bound check; `snprintf`/`strerror_r` buffer sizing; copying a C string of
  unknown length without the `MAX_STR_LENGTH` cap; reading `n` bytes from a pointer where `n` is
  attacker-influenced; unaligned stores writing past the intended cell.
- **Format-string safety.** `printf`/`fprintf`/`snprintf`/`sscanf`: is the format validated against
  the argument types at typing time? Is `%n` rejected (including BSD `q` and Microsoft `I32`/`I64`
  length modifiers)? Can a caller pass a runtime (non-literal) format that smuggles a conversion?
- **SQL handling.** `query_to_array` / `sqlite3_exec` run caller-supplied SQL — is that the documented
  contract (caller trusts its own SQL), or is there an injection surface where numbox interpolates
  untrusted data into SQL text? Parameter binding vs string concatenation.
- **NUL / encoding.** TEXT/BLOB NUL handling (interior NUL truncation = silent data loss/disclosure);
  UTF-8/UTF-32 validation on the typemap path (does invalid UTF-8 get rejected or does it write
  malformed/over-long bytes?); embedded-NUL in a path or SQL string.
- **Library / path loading.** `_resolve_lib_path`, `_windows_bundled_dll_path` (CPython `DLLs/`, conda
  `Library/bin/`), `RTLD_GLOBAL` loading: is the search order safe (no attacker-writable dir ahead of
  the system lib; no relative-path / cwd DLL-planting risk on Windows)? Are addresses read from the
  *intended* library handle (the macOS `add_symbol` fix) rather than an arbitrary one?
- **Integer issues with a safety consequence.** Size/length computed by arithmetic that can overflow
  or go negative and then feeds an allocation, a memcpy length, or a pointer offset.
- **Unsafe defaults.** `SQLITE_STATIC` where the buffer does not outlive the statement (use-after-free
  surfacing as a disclosure of freed memory); a destructor sentinel passed wrong.

## Calibration

- Tie every SEC finding to a concrete **consequence** (corruption / crash / disclosure / wrong-trust),
  not "unvalidated input" in the abstract. If the input is trusted-by-contract (a library API the
  caller controls), say so and downgrade or omit.
- Distinguish a real reachable hazard (high/critical) from defense-in-depth hardening (low/medium).
