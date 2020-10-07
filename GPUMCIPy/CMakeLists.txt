include_directories ("${PROJECT_SOURCE_DIR}")

if (WIN32)
	set(FILESTART "")
	set(FILETYPE "dll")
	set(TARGETFILETYPE "pyd")
	set(TARGET_PY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>")
else(WIN32)
	set(FILESTART "lib")
	set(FILETYPE "so")
	set(TARGETFILETYPE "so")
	set(TARGET_PY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif(WIN32)


set(LIB_NAME "GPUMCIPy")
set(PYTHON_LIBS ${PYTHON_LIBS} ${LIB_NAME})
cuda_include_directories ("${PROJECT_SOURCE_DIR}")
CUDA_ADD_LIBRARY(${LIB_NAME} SHARED "GPUMCIPy.cpp")
# CUDA_ADD_CUBLAS_TO_TARGET( ${LIB_NAME} ) possibly needed
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER python)
add_dependencies(${LIB_NAME} odlcpputils_cuda)
target_link_libraries(${LIB_NAME} implementations odlcpputils_cuda ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

#TODO: LINUX?
set(INSTALL_LIB "pyinstall")
add_custom_target(${INSTALL_LIB} COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${TARGET_PY}/__init__.py)
set_property(TARGET ${INSTALL_LIB} PROPERTY FOLDER python)

# do the copying
foreach(file_i ${PYTHON_LIBS})
    add_custom_command(TARGET ${INSTALL_LIB} COMMAND ${CMAKE_COMMAND} -E copy ${TARGET_PY}/${FILESTART}${file_i}.${FILETYPE} ${TARGET_PY}/${file_i}.${TARGETFILETYPE})
endforeach()

add_custom_command(TARGET ${INSTALL_LIB} COMMAND cd ${TARGET_PY} && python ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install)
									
add_dependencies(${INSTALL_LIB} ${PYTHON_LIBS})
