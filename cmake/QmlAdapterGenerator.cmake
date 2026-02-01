# Generates Qt/QML adapter headers from config headers, and wires the generated
# headers into one or more targets.

find_package(Python REQUIRED)

function(add_qml_adapters)
    set(options "")
    set(oneValueArgs TEMPLATES_DIR OUT_DIR)
    set(multiValueArgs TARGETS CONFIG_INPUTS)
    cmake_parse_arguments(ADAPTER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ADAPTER_TARGETS)
        message(FATAL_ERROR "add_qml_adapters: TARGETS must be specified")
    endif()

    if(NOT ADAPTER_CONFIG_INPUTS)
        return()
    endif()

    if(NOT ADAPTER_TEMPLATES_DIR)
        set(ADAPTER_TEMPLATES_DIR "${CMAKE_SOURCE_DIR}/src/templates")
    endif()

    if(NOT ADAPTER_OUT_DIR)
        set(ADAPTER_OUT_DIR "${CMAKE_BINARY_DIR}/generated/qml_adapters")
    endif()

    if(NOT EXISTS "${ADAPTER_TEMPLATES_DIR}")
        message(FATAL_ERROR "add_qml_adapters: TEMPLATES_DIR does not exist: ${ADAPTER_TEMPLATES_DIR}")
    endif()

    # Track template files as dependencies (to re-render when templates change)
    file(GLOB ADAPTER_TEMPLATE_FILES "${ADAPTER_TEMPLATES_DIR}/*.j2")

    set(CONFIG_INPUTS  "")
    set(CONFIG_OUTPUTS "")

    foreach(config_header IN LISTS ADAPTER_CONFIG_INPUTS)
        get_filename_component(CONFIG_INPUT "${config_header}" ABSOLUTE)

        file(RELATIVE_PATH REL_PATH "${CMAKE_SOURCE_DIR}" "${CONFIG_INPUT}")
        string(REPLACE "\\" "/" REL_PATH "${REL_PATH}")

        # Strip leading "src/" so the output include root matches <session/...>
        string(REGEX REPLACE "^src/" "" REL_NO_SRC "${REL_PATH}")

        # Output dir mirrors the header's relative directory.
        get_filename_component(REL_DIR  "${REL_NO_SRC}" DIRECTORY)
        get_filename_component(REL_BASE "${REL_NO_SRC}" NAME_WE)

        # If base ends with "device_config", strip trailing "_config",
        # because the template produces an adapter class that extends beyond just
        # configuration params, it is a "controller" from QML too
        set(REL_BASE_FOR_ADAPTER "${REL_BASE}")
        if(REL_BASE_FOR_ADAPTER MATCHES "device_config$")
            string(REGEX REPLACE "_config$" "" REL_BASE_FOR_ADAPTER "${REL_BASE_FOR_ADAPTER}")
        endif()

        set(CONFIG_OUTPUT "${ADAPTER_OUT_DIR}/${REL_DIR}/${REL_BASE_FOR_ADAPTER}_adapter.gen.h")

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
                --templates-dir "${ADAPTER_TEMPLATES_DIR}"
                --out-dir "${ADAPTER_OUT_DIR}"
                --src-root "${CMAKE_SOURCE_DIR}"
                ${CONFIG_INPUTS}
        DEPENDS
            ${CONFIG_INPUTS}
            "${GENERATOR_SCRIPT}"
            ${ADAPTER_TEMPLATE_FILES}
        COMMENT "Generating Qt/QML adapters"
        VERBATIM
    )

    # Link the generated stuff to our target(s)
    foreach(target IN LISTS ADAPTER_TARGETS)
        target_include_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/src")
        target_include_directories(${target} PUBLIC "${ADAPTER_OUT_DIR}")
        target_include_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/src/ui")

        target_sources(${target} PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ui/models/config_adapter.h
            ${CMAKE_SOURCE_DIR}/src/ui/models/device_adapter.h
            ${CONFIG_OUTPUTS}
        )

        target_link_libraries(${target} PRIVATE Qt6::Core)
    endforeach()
endfunction()
