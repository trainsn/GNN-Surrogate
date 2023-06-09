cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(MPASPermBack)

set(CMAKE_CXX_STANDARD 11)

# Add .lib files
link_directories(
    "${CMAKE_SOURCE_DIR}/lib"
)

# Add source files
file(GLOB_RECURSE SOURCE_FILES 
	${CMAKE_SOURCE_DIR}/src/*.c
	${CMAKE_SOURCE_DIR}/src/*.cpp)
	
# Add header files
file(GLOB_RECURSE HEADER_FILES 
	${CMAKE_SOURCE_DIR}/src/*.h
	${CMAKE_SOURCE_DIR}/src/*.hpp)
	
# We need a CMAKE_DIR with some code to find external dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

#######################################
# LOOK for the packages that we need! #
#######################################

# NETCDF
link_directories(/usr/local/netcdf/intel/17.0/mvapich2/2.3/4.6.1/lib)
link_libraries(netcdf)

# MPI 
link_directories(/opt/mvapich2/intel/17.0/2.3.2/lib)
link_libraries(mpi)

# Define the include DIRs
include_directories(
	"${CMAKE_SOURCE_DIR}/src"
	"${CMAKE_SOURCE_DIR}/include"
	"/usr/local/netcdf/intel/17.0/mvapich2/2.3/4.6.1/include"
	"/opt/mvapich2/intel/17.0/2.3.2/include"
	"/users/PAS0027/trainsn/software/eigen3"
)

# Define the executable
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

# Put all libraries into a variable
set(LIBS netcdf mpi)

# Define the link libraries
target_link_libraries(${PROJECT_NAME} ${LIBS})

