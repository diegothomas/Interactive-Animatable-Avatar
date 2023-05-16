# CMake entry point
cmake_minimum_required (VERSION 3.0)


if (_APPLE_)
	project(MATCHSMPL LANGUAGES CXX C)

	SET(CMAKE_CXX_FLAGS "-O3 -g -std=c++11  -Wno-c++11-narrowing ${CMAKE_CXX_FLAGS}")
	find_package(Eigen3 REQUIRED)

	if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
        		message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever 			name, like bin_Visual2012_64bits/)" )
    	endif()
    	if( CMAKE_SOURCE_DIR MATCHES " " )
        		message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
    	endif()
    	if( CMAKE_BINARY_DIR MATCHES " " )
        		message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
    	endif()


	include_directories(
        		${EIGEN3_INCLUDE_DIR}
        		/usr/include/
		../
        		.
	)

	add_definitions(
        		-DTW_STATIC
        		-DTW_NO_LIB_PRAGMA
        		-DTW_NO_DIRECT3D
        		-DGLEW_STATIC
        		-D_CRT_SECURE_NO_WARNINGS
        		-DGL_SILENCE_DEPRECATION
        		-D__gl3_h_
        		-DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
		-D_APPLE_
    	)

	add_executable(matchsmpl
        		MatchSMPL.cpp
        		../src/Utilities.cpp
        		../include/Utilities.h
        		../include/MeshUtils.h
    	)
    
    	target_link_libraries(matchsmpl
        		${ALL_LIBS}
    	)

else()
	project(MATCHSMPL LANGUAGES CXX C)

	SET(CMAKE_CXX_FLAGS "-O3 -g -std=c++11 ${CMAKE_CXX_FLAGS}")
	SET(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../external/eigen-3.3.9")

	if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
        		message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever 			name, like bin_Visual2012_64bits/)" )
    	endif()
    	if( CMAKE_SOURCE_DIR MATCHES " " )
        		message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
    	endif()
    	if( CMAKE_BINARY_DIR MATCHES " " )
        		message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
    	endif()

	# On Visual 2005 and above, this module can set the debug working directory
    	cmake_policy(SET CMP0026 OLD)
    	list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../external/rpavlik-cmake-modules-fe2273")
    	include(CreateLaunchers)
    	include(MSVCMultipleProcessCompile) # /MP


	include_directories(
        		${EIGEN3_INCLUDE_DIR}
        		/usr/include/
		../
        		.
	)

	add_definitions(
        		-DTW_STATIC
        		-DTW_NO_LIB_PRAGMA
        		-DTW_NO_DIRECT3D
        		-DGLEW_STATIC
        		-D_CRT_SECURE_NO_WARNINGS
        		-DGL_SILENCE_DEPRECATION
        		-D__gl3_h_
        		-DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
    	)

	add_executable(matchsmpl
        		MatchSMPL.cpp
        		../src/Utilities.cpp
        		../include/Utilities.h
        		../include/MeshUtils.h
    	)
    
    	target_link_libraries(matchsmpl
        		${ALL_LIBS}
    	)

endif()


