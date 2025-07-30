idf_component_register(
    SRCS "mpu6050.cpp" "mpu6050_hal.cpp"
    INCLUDE_DIRS "."
    REQUIRES driver
)