# libsystemd and dbus are required to build the tracy profiler
# but both fail on gcc 15, use the system packages for them

set(VCPKG_POLICY_EMPTY_PACKAGE enabled)
