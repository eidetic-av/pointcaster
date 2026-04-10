
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (UNIX)
    add_custom_target(link_compile_commands ALL
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
    )
else()
    # windows needs to make a junction link as opposed to a symlink
    # in order to allow cmake to run as an unprivelidged user

    # TODO still not working, maybe need to just copy the file?

    # file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}" native_bin_dir)
    # file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}" native_src_dir)
    # add_custom_target(link_compile_commands ALL 
    #     COMMAND cmd.exe /c mklink /J 
    #     "${native_src_dir}\\compile_commands.json" 
    #     "${native_bin_dir}\\compile_commands.json" 
    # )
endif()