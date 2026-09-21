# CARLA 0.9.15 dormant sensor null-stream guard

Experimental process-local workaround, limited to SHA256 `1da989e4ff136e0658fdea421bc1da2b0e5cb018c6e767efc4f6129c21ab1a81`.
The original simulator executable is never modified. The wrapper fails closed for any other binary.

GDB diagnostic: route23670, 71.35 sim seconds, SIGSEGV in token conversion via ASensor::EndPlay, FActorRegistry::PutActorToSleep, LargeMapManager::ConvertActiveToDormantActors.
Official 0.9.15 ActorData.cpp moves the sensor stream to dormant storage before destroying the active actor. Sensor.cpp then unconditionally reads/ closes its moved-from stream. Optional IsStreamReady alone may remain true after moving the underlying shared_ptr; guard the actual shared state.

At 0x27d0956 the deployed binary has just loaded the shared state into rsi. For null state only, jump to 0x27d0980 (sensor-manager deregistration), retaining Super::EndPlay, normal actor dormancy, stream stored for wakeup, and the entire nonnull cleanup path. Log each intervention without unwinding frames. GDB 12.1 on this binary asserts while unwinding deep DWARF frames; the diagnostic stack is already archived. Do not skip the entire EndPlay function, alter actor-active distance, remove traffic, disable physics, or change score/termination rules.

This is an instrumented simulator variant. Report it with each result; do not claim bitwise replay equivalence to historical CARLA. Prefer a rebuilt source-level null-state guard for long-term deployment; no global install or simulator upgrade here.

Sources:
- https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Actor/ActorData.cpp#L235-L247
- https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/Sensor.cpp#L105-L120
- https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/streaming/detail/Stream.h#L32-L34

## Opt-in use

```sh
export CARLA_ORIGINAL_ROOT=/path/to/the/verified/carla-installation
export CARLA_ROOT="$PWD/tools/carla_sensor_guard"
# Launch the normal evaluator; it will invoke $CARLA_ROOT/CarlaUE4.sh.
```

Keep CARLA PythonAPI/PYTHONPATH pointing to the original installation. Requires
GNU gdb and sha256sum. GDB can select a debugger executable (default: gdb).
No evaluator, training config, or default CARLA_ROOT enables this workaround.
Removing these environment overrides returns to the original launcher.

The null branch was exercised at 69.15 simulated seconds on route23670 and
continued beyond the previous crash interval. This establishes the workaround
for that exact failure, not general simulator correctness or improved driving.
The original first diagnostic and first guard attempt hit a separate GDB DWARF
unwind assertion; this version intentionally does not print runtime backtraces.
Research evidence: RockWenJJ/A100_DriveTransformer_Research commit efac44a,
research/reports/s32_crash_s16_eval_20260921/.
