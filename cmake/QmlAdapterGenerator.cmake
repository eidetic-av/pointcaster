find_package(Python REQUIRED)

function(add_qml_adapters)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs TARGETS CONFIG_INPUTS)
    cmake_parse_arguments(ADAPTER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ADAPTER_TARGETS)
        message(FATAL_ERROR "add_qml_adapters: TARGETS must be specified")
    endif()

    if(NOT ADAPTER_CONFIG_INPUTS)
        return()
    endif()

    set(CONFIG_INPUTS  "")
    set(CONFIG_OUTPUTS "")

    foreach(config_header IN LISTS ADAPTER_CONFIG_INPUTS)
        get_filename_component(CONFIG_INPUT "${config_header}" ABSOLUTE)
        get_filename_component(CONFIG_DIR   "${CONFIG_INPUT}" DIRECTORY)
        get_filename_component(CONFIG_NAME  "${CONFIG_INPUT}" NAME_WE)

        set(CONFIG_OUTPUT "${CONFIG_DIR}/${CONFIG_NAME}.gen.h")

        list(APPEND CONFIG_INPUTS  "${CONFIG_INPUT}")
        list(APPEND CONFIG_OUTPUTS "${CONFIG_OUTPUT}")
    endforeach()

    if(NOT CONFIG_INPUTS)
        return()
    endif()

    set(GENERATOR_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/generate-qt-adapters.py")

    add_custom_command(
        OUTPUT ${CONFIG_OUTPUTS}
        COMMAND "${Python_EXECUTABLE}"
                "${GENERATOR_SCRIPT}"
                ${CONFIG_INPUTS}
        DEPENDS
            ${CONFIG_INPUTS}
            "${GENERATOR_SCRIPT}"
        COMMENT "Generating Qt/QML adapters"
        VERBATIM
    )

    foreach(target IN LISTS ADAPTER_TARGETS)
        target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src/ui)
        target_sources(${target} PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ui/models/config_adapter.h
            ${CONFIG_OUTPUTS}
        )
        target_link_libraries(${target} PRIVATE Qt6::Core)
    endforeach()
endfunction()
