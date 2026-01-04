file(GLOB_RECURSE TO_DELETE
  "${ROOT}/MSVCP*.dll"
  "${ROOT}/CONCRT*.dll"
  "${ROOT}/VCRUNTIME*.dll"
)
if(TO_DELETE)
  file(REMOVE ${TO_DELETE})
endif()
