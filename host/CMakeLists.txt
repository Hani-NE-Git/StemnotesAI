cmake_minimum_required(VERSION 3.22)
project(StemNotesHost VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# JUCE: use a local checkout (JUCE_DIR) or fetch it
if(JUCE_DIR)
  add_subdirectory(${JUCE_DIR} juce EXCLUDE_FROM_ALL)
else()
  include(FetchContent)
  FetchContent_Declare(juce GIT_REPOSITORY https://github.com/juce-framework/JUCE GIT_TAG 8.0.4 GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(juce)
endif()

juce_add_console_app(stemnotes-host PRODUCT_NAME "stemnotes-host")
target_sources(stemnotes-host PRIVATE src/Main.cpp src/Engine.cpp src/Commands.cpp)
target_include_directories(stemnotes-host PRIVATE third_party src)
target_compile_definitions(stemnotes-host PRIVATE
  JUCE_PLUGINHOST_VST3=1
  JUCE_PLUGINHOST_VST=0
  JUCE_PLUGINHOST_AU=0
  JUCE_PLUGINHOST_LADSPA=0
  JUCE_USE_MP3AUDIOFORMAT=1
  JUCE_USE_OGGVORBIS=1
  JUCE_USE_FLAC=1
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
  JUCE_MODAL_LOOPS_PERMITTED=1
  JUCE_ASIO=0
  JUCE_WASAPI=1
  JUCE_DIRECTSOUND=1
  JUCE_ALSA=1
  JUCE_JACK=0
  JUCE_STANDALONE_APPLICATION=1
  STEMNOTES_HOST_VERSION="${PROJECT_VERSION}")
target_link_libraries(stemnotes-host PRIVATE
  juce::juce_audio_utils
  juce::juce_audio_processors
  juce::juce_audio_formats
  juce::juce_audio_devices
  juce::juce_gui_basics
  juce::juce_recommended_config_flags
  juce::juce_recommended_lto_flags)
if(WIN32)
  set_target_properties(stemnotes-host PROPERTIES WIN32_EXECUTABLE FALSE)
endif()
