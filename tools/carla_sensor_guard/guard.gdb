set pagination off
set confirm off
set backtrace limit 1
handle SIGPIPE nostop noprint pass
handle SIG32 nostop noprint pass
handle SIG33 nostop noprint pass
# This exact address is guarded by the launcher's binary SHA256 check.
# ASensor::EndPlay: rsi <- this->Stream._shared_state (at 0x27d094e).
# A dormant sensor has moved this shared_ptr to FActorSensorData.
# Skip token lookup + CloseStream only when the shared_ptr is null.
# Continue at sensor-manager deregistration; Super::EndPlay already ran.
break *0x27d0956
condition 1 $rsi == 0
commands 1
silent
printf "CARLA_SENSOR_NULL_STREAM_GUARD sensor=%p state=%p\n", $r12, $rsi
set $pc = 0x27d0980
continue
end
run
quit
