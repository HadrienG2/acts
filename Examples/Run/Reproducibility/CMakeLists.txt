add_executable(
  ActsParallelReproducibilityTest
  ParallelReproducibilityTest.cpp
  ReadRootFile.cpp)
target_link_libraries(
  ActsParallelReproducibilityTest
  PRIVATE ROOT::Core ROOT::RIO ROOT::Tree ROOT::TreePlayer)

install(
  TARGETS ActsParallelReproducibilityTest
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
