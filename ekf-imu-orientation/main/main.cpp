/*
 * Copyright (c) 2025, Mezael Docoy
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>
#include <memory>
#include <string>
#include <sstream>
#include <esp_pthread.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_log.h>
#include <cmath>
#include <Eigen/Dense>

#include "mpu6050.hpp"

using namespace std::chrono_literals;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const char* TAG = "main";
static std::atomic<float> alpha{0.98f}; // Complementary filter coefficient

// Orientation estimates
struct Orientation {
    float roll;
    float pitch;
    float yaw;

    Orientation() : roll(0.0f), pitch(0.0f), yaw(0.0f) {}
    Orientation(float r, float p, float y) : roll(r), pitch(p), yaw(y) {}
};

static std::atomic<Orientation> comp_filter_orientation{Orientation{0.0f, 0.0f, 0.0f}};
static std::atomic<Orientation> ekf_orientation{Orientation{0.0f, 0.0f, 0.0f}};

static MPU6050::Device Dev = {
    .i2cPort = 0,
    .i2cAddress = MPU6050::I2C_ADDRESS_MPU6050_AD0_L
};

// Extended Kalman Filter implementation
class OrientationEKF {
private:
    // State vector: [roll, pitch, yaw, gyro_bias_x, gyro_bias_y, gyro_bias_z] (radians)
    Eigen::VectorXf x;

    // Covariance matrix
    Eigen::MatrixXf P;

    // Process noise covariance
    Eigen::MatrixXf Q;

    // Measurement noise covariance
    Eigen::MatrixXf R;

    // Time step
    float dt;

    // Gravity constant
    const float g = 9.81f;

    // For covariance conditioning
    const float MIN_COVARIANCE = 1e-6f;
    const float MAX_COVARIANCE = 1e6f;
    const float MAX_STATE_VALUE = 100.0f * M_PI;

    void conditionCovariance() {
        P = (P + P.transpose()) * 0.5f;
        for(int i=0; i<P.rows(); i++) {
            P(i,i) = std::max(P(i,i), MIN_COVARIANCE);
            for(int j=0; j<P.cols(); j++) {
                P(i,j) = std::min(std::max(P(i,j), -MAX_COVARIANCE), MAX_COVARIANCE);
            }
        }
    }

public:
    OrientationEKF() {
        reset();
        dt = 0.02f;
    }

    void reset() {
        x = Eigen::VectorXf::Zero(6);
        P = Eigen::MatrixXf::Identity(6, 6) * 0.1f;

        Q = Eigen::MatrixXf::Identity(6, 6);
        Q(0,0) = Q(1,1) = Q(2,2) = 0.01f;
        Q(3,3) = Q(4,4) = Q(5,5) = 0.001f;

        R = Eigen::MatrixXf::Identity(3, 3) * 0.1f;
    }

    void setTimeStep(float delta_t) {
        dt = delta_t;
    }

    void predict(const Eigen::Vector3f& gyro_rates) {
        if(x.array().abs().maxCoeff() > MAX_STATE_VALUE || 
           P.array().abs().maxCoeff() > MAX_COVARIANCE*10) {
            ESP_LOGW(TAG, "EKF diverged in predict! Resetting...");
            reset();
            return;
        }

        Eigen::VectorXf x_pred(6);

        float roll = x(0);
        float pitch = x(1);
        float yaw = x(2);
        float g_bias_x = x(3);
        float g_bias_y = x(4);
        float g_bias_z = x(5);

        const float pitch_threshold = 85.0f * M_PI/180.0f;
        if(abs(pitch) > pitch_threshold) {
            pitch = std::copysign(pitch_threshold, pitch);
        }

        // Corrected gyro measurements
        float wx = gyro_rates(0) - g_bias_x;
        float wy = gyro_rates(1) - g_bias_y;
        float wz = gyro_rates(2) - g_bias_z;

        // Euler integration (gyro in rad/s)
        x_pred(0) = roll + dt * (wx + sin(roll) * tan(pitch) * wy + cos(roll) * tan(pitch) * wz);
        x_pred(1) = pitch + dt * (cos(roll) * wy - sin(roll) * wz);
        x_pred(2) = yaw + dt * (sin(roll) / cos(pitch) * wy + cos(roll) / cos(pitch) * wz);

        x_pred(3) = g_bias_x;
        x_pred(4) = g_bias_y;
        x_pred(5) = g_bias_z;

        // Jacobian
        Eigen::MatrixXf F = Eigen::MatrixXf::Identity(6, 6);
        F(0,3) = -dt;
        F(1,4) = -dt * cos(roll);
        F(1,5) = dt * sin(roll);

        P = F * P * F.transpose() + Q;

        conditionCovariance();
        x = x_pred;
    }

    void update(const Eigen::Vector3f& accel) {
        float accel_norm = accel.norm();
        if(accel_norm < 6.0f || accel_norm > 13.0f) {
            ESP_LOGD(TAG, "Rejected bad accel data: norm=%.2f", accel_norm);
            return;
        }

        if(x.array().abs().maxCoeff() > MAX_STATE_VALUE || 
           P.array().abs().maxCoeff() > MAX_COVARIANCE*10) {
            ESP_LOGW(TAG, "EKF diverged in update! Resetting...");
            reset();
            return;
        }

        // Predicted gravity vector
        Eigen::Vector3f z_pred;
        z_pred(0) = -sin(x(1)) * g;
        z_pred(1) = sin(x(0)) * cos(x(1)) * g;
        z_pred(2) = cos(x(0)) * cos(x(1)) * g;

        Eigen::Vector3f y = accel - z_pred;

        Eigen::MatrixXf H = Eigen::MatrixXf::Zero(3, 6);
        H(0,1) = -cos(x(1)) * g;
        H(1,0) = cos(x(0)) * cos(x(1)) * g;
        H(1,1) = -sin(x(0)) * sin(x(1)) * g;
        H(2,0) = -sin(x(0)) * cos(x(1)) * g;
        H(2,1) = -cos(x(0)) * sin(x(1)) * g;

        Eigen::MatrixXf S = H * P * H.transpose() + R;

        Eigen::FullPivLU<Eigen::MatrixXf> lu(S);
        if(!lu.isInvertible()) {
            ESP_LOGE(TAG, "Matrix inversion failed. Resetting...");
            reset();
            return;
        }

        Eigen::MatrixXf K = P * H.transpose() * S.inverse();

        x = x + K * y;

        // Joseph form update for stability
        Eigen::MatrixXf I = Eigen::MatrixXf::Identity(6, 6);
        P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

        conditionCovariance();
    }

    Orientation getOrientation() const {
        float roll = fmod(static_cast<float>(x(0) * 180.0f / M_PI), 360.0f);
        float pitch = fmod(static_cast<float>(x(1) * 180.0f / M_PI), 360.0f);
        float yaw = fmod(static_cast<float>(x(2) * 180.0f / M_PI), 360.0f);

        if(roll > 180.0f) roll -= 360.0f;
        if(pitch > 180.0f) pitch -= 360.0f;
        if(yaw > 180.0f) yaw -= 360.0f;

        return {roll, pitch, yaw};
    }
};

Orientation computeAccelOrientation(float ax, float ay, float az) {
    Orientation o;
    o.roll = atan2f(ay, az) * 180.0f / M_PI;
    o.pitch = atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / M_PI;
    o.yaw = 0.0f;
    return o;
}

Orientation integrateGyroOrientation(const Orientation& prev, 
                                   float gx, float gy, float gz, 
                                   float dt) {
    Orientation o;

    float gx_rad = gx * M_PI / 180.0f;
    float gy_rad = gy * M_PI / 180.0f;
    float gz_rad = gz * M_PI / 180.0f;

    float prev_roll = prev.roll * M_PI / 180.0f;
    float prev_pitch = prev.pitch * M_PI / 180.0f;
    float prev_yaw = prev.yaw * M_PI / 180.0f;

    const float pitch_threshold = 85.0f * M_PI/180.0f;
    if(abs(prev_pitch) > pitch_threshold) {
        prev_pitch = std::copysign(pitch_threshold, prev_pitch);
    }

    o.roll = prev_roll + dt * (gx_rad + sin(prev_roll) * tan(prev_pitch) * gy_rad + 
                          cos(prev_roll) * tan(prev_pitch) * gz_rad);
    o.pitch = prev_pitch + dt * (cos(prev_roll) * gy_rad - sin(prev_roll) * gz_rad);
    o.yaw = prev_yaw + dt * (sin(prev_roll) / cos(prev_pitch) * gy_rad + 
                        cos(prev_roll) / cos(prev_pitch) * gz_rad);

    o.roll *= 180.0f / M_PI;
    o.pitch *= 180.0f / M_PI;
    o.yaw *= 180.0f / M_PI;

    o.roll = fmod(o.roll, 360.0f);
    o.pitch = fmod(o.pitch, 360.0f);
    o.yaw = fmod(o.yaw, 360.0f);

    if(o.roll > 180.0f) o.roll -= 360.0f;
    if(o.pitch > 180.0f) o.pitch -= 360.0f;
    if(o.yaw > 180.0f) o.yaw -= 360.0f;

    return o;
}

bool calibrateSensors(MPU6050::MPU6050_Driver& mpu, 
                     Eigen::Vector3f& gyroBias, 
                     int samples = 500) {
    Eigen::Vector3f sumGyro = Eigen::Vector3f::Zero();
    int successCount = 0;

    ESP_LOGI(TAG, "Calibrating IMU...");

    for (int i = 0; i < samples; i++) {
        MPU6050::Mpu6050_GyroData_t gyroData;
        if (mpu.Mpu6050_GetGyroData(gyroData) == Mpu6050_Error_t::MPU6050_OK) {
            sumGyro(0) += gyroData.Gyro_X;
            sumGyro(1) += gyroData.Gyro_Y;
            sumGyro(2) += gyroData.Gyro_Z;
            successCount++;
        }
        std::this_thread::sleep_for(10ms);
    }

    if (successCount < samples * 0.9) {
        ESP_LOGE(TAG, "Calibration failed - too many read errors");
        return false;
    }

    gyroBias = sumGyro / successCount;
    ESP_LOGI(TAG, "Calibration complete. Gyro bias (X,Y,Z): %.2f, %.2f, %.2f °/s", 
             gyroBias(0), gyroBias(1), gyroBias(2));
    return true;
}

void sendGains() {
    std::cout << "Gains," << alpha.load() << std::endl;
}

void gain_tuning_thread() {
    char buffer[64];

    std::this_thread::sleep_for(1000ms);
    sendGains();

    while (true) {
        if (fgets(buffer, sizeof(buffer), stdin) != nullptr) {
            char type;
            float value;
            if (sscanf(buffer, " %c %f", &type, &value) == 2) {
                switch (type) {
                    case 'a':
                    case 'A':
                        alpha = value;
                        ESP_LOGI(TAG, "Complimentary filter alpha set to: %.2f", alpha.load());
                        break;
                    default:
                        break;
                }
            } else {
                ESP_LOGD(TAG, "Invalid input format: %s", buffer);
            }
        }

        std::this_thread::sleep_for(10ms);
    }
}

void imu_sensor_thread() {
    if (mpu6050_hal_init(Dev.i2cPort) == Mpu6050_Error_t::MPU6050_ERR) {
        ESP_LOGE(TAG, "Failed to initialize I2C HAL");
        return;
    }

    MPU6050::MPU6050_Driver mpu(Dev);
    if (mpu.Mpu6050_Init(&MPU6050::DefaultConfig) != Mpu6050_Error_t::MPU6050_OK) {
        ESP_LOGE(TAG, "MPU6050 initialization failed!");
        return;
    }

    uint8_t dev_id = 0;
    if (mpu.Mpu6050_GetDevideId(dev_id) != Mpu6050_Error_t::MPU6050_OK || dev_id != MPU6050::WHO_AM_I_VAL) {
        ESP_LOGE(TAG, "Invalid MPU6050 device ID: 0x%x", dev_id);
        return;
    }

    ESP_LOGI(TAG, "MPU6050 initialized successfully. Device ID: 0x%x", dev_id);

    Eigen::Vector3f gyroBias = Eigen::Vector3f::Zero();
    if (!calibrateSensors(mpu, gyroBias)) {
        ESP_LOGW(TAG, "Failed calibration, using zero bias");
    }

    std::this_thread::sleep_for(500ms);

    auto prevTime = std::chrono::steady_clock::now();
    int errorCount = 0;
    const int maxErrorCount = 10;

    OrientationEKF ekf;

    while (true) {
        MPU6050::Mpu6050_AccelData_t accelData;
        MPU6050::Mpu6050_GyroData_t gyroData;

        if (mpu.Mpu6050_GetAccelData(accelData) != Mpu6050_Error_t::MPU6050_OK ||
            mpu.Mpu6050_GetGyroData(gyroData) != Mpu6050_Error_t::MPU6050_OK) {
            if (++errorCount > maxErrorCount) {
                ESP_LOGE(TAG, "Too many sensor errors, exiting!");
                return;
            }
            std::this_thread::sleep_for(20ms);
            continue;
        }
        errorCount = 0;

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - prevTime).count();
        prevTime = now;

        Eigen::Vector3f gyroRates;
        gyroRates(0) = (gyroData.Gyro_X - gyroBias(0)) * M_PI / 180.0f;
        gyroRates(1) = (gyroData.Gyro_Y - gyroBias(1)) * M_PI / 180.0f;
        gyroRates(2) = (gyroData.Gyro_Z - gyroBias(2)) * M_PI / 180.0f;

        Eigen::Vector3f accel;
        accel(0) = accelData.Accel_X * 9.81f;
        accel(1) = accelData.Accel_Y * 9.81f;
        accel(2) = accelData.Accel_Z * 9.81f;

        bool accel_valid = (accel.norm() > 6.0f && accel.norm() < 13.0f);

        if(accel_valid) {
            accel.normalize();
            accel *= 9.81f;
        }

        Orientation accOrientation = {0, 0, 0};
        if(accel_valid) {
            accOrientation = computeAccelOrientation(accelData.Accel_X, 
                                                   accelData.Accel_Y, 
                                                   accelData.Accel_Z);
        }

        Orientation gyroOrientation = integrateGyroOrientation(comp_filter_orientation.load(), 
                                                             gyroRates(0) * 180.0f/M_PI, 
                                                             gyroRates(1) * 180.0f/M_PI, 
                                                             gyroRates(2) * 180.0f/M_PI, 
                                                             dt);

        Orientation compOrientation = gyroOrientation;
        if(accel_valid) {
            compOrientation.roll = alpha * gyroOrientation.roll + (1.0f - alpha) * accOrientation.roll;
            compOrientation.pitch = alpha * gyroOrientation.pitch + (1.0f - alpha) * accOrientation.pitch;
        }
        comp_filter_orientation = compOrientation;

        ekf.setTimeStep(dt);
        ekf.predict(gyroRates);
        if(accel_valid) {
            ekf.update(accel);
        }

        ekf_orientation = ekf.getOrientation();

        Orientation comp = comp_filter_orientation.load();
        Orientation ekfO = ekf_orientation.load();

        printf("CF,%.2f,%.2f,%.2f,EKF,%.2f,%.2f,%.2f\n",
               comp.roll, comp.pitch, comp.yaw,
               ekfO.roll, ekfO.pitch, ekfO.yaw);

        std::this_thread::sleep_for(20ms);
    }
}

extern "C" void app_main(void) {
    esp_pthread_cfg_t cfg = esp_pthread_get_default_config();
    cfg.stack_size = 8192; // Increased stack size for EKF
    cfg.prio = 5;
    cfg.pin_to_core = 0;
    cfg.thread_name = "imu_thread";
    ESP_ERROR_CHECK(esp_pthread_set_cfg(&cfg));

    std::thread imu_thread(imu_sensor_thread);
    imu_thread.detach();

    esp_pthread_cfg_t cfg_gain = esp_pthread_get_default_config();
    cfg_gain.stack_size = 4096;
    cfg_gain.prio = 5;
    cfg_gain.pin_to_core = 1;
    cfg_gain.thread_name = "gain_thread";
    ESP_ERROR_CHECK(esp_pthread_set_cfg(&cfg_gain));
    std::thread gain_thread(gain_tuning_thread);
    gain_thread.detach();
}