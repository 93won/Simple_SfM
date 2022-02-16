add_library(sfm SHARED
types/Camera.cpp
types/Feature.cpp
types/Frame.cpp
types/Map.cpp
types/MapPoint.cpp
utils/Viewer.cpp
utils/Config.cpp
core/SfM.cpp)

target_link_libraries(sfm ${THIRD_PARTY_LIBS})
