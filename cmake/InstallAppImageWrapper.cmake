set(bin "${CMAKE_INSTALL_PREFIX}/bin")

# copy the launcher in its place and make it executable
file(COPY "${POINTCASTER_SOURCE_DIR}/packaging/pointcaster.sh"
     DESTINATION "${bin}"
     FILE_PERMISSIONS
       OWNER_READ OWNER_WRITE OWNER_EXECUTE
       GROUP_READ GROUP_EXECUTE
       WORLD_READ WORLD_EXECUTE)