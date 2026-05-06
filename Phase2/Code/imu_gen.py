import numpy as np
from ImuUtils import acc_gen, gyro_gen, vib_from_env, gyro_low_accuracy, accel_low_accuracy 


def generate_imu_data_np(ref_a, ref_w, fs):

    # sets random vibration to accel with RMS for x/y/z axis - 1/2/3 m/s^2, can be zero or changed to other values
    acc_env = '[0.03 0.001 0.01]-random'
    acc_vib_def = vib_from_env(acc_env, fs)

    real_acc = acc_gen(fs, ref_a, accel_low_accuracy, acc_vib_def)


    # sets sinusoidal vibration to gyro with frequency being 0.5 Hz and amp for x/y/z axis being 6/5/4 deg/s
    gyro_env = '[6 5 4]d-0.5Hz-sinusoidal'
    gyro_vib_def = vib_from_env(gyro_env, fs)

    real_gyro = gyro_gen(fs, ref_w, gyro_low_accuracy, gyro_vib_def)

    imu_data = np.concat([real_acc, real_gyro], axis=1)

    return imu_data