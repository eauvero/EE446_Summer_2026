#include <Arduino_HS300x.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_APDS9960.h>
#include <math.h>

bool isInitialized = false;

float ALPHA = 0.15;
unsigned long last_baseline_TriggerTime = 0;
unsigned long last_breathWarmAir_TriggerTime = 0;
unsigned long last_magneticDisturbance_TriggerTime = 0;
unsigned long last_lightColorChange_TriggerTime = 0;
const unsigned long eventTriggerLockOutDuration = 3000;
const unsigned long baselineTriggerLockOutDuration = 6000;


float RH_THRESHOLD = 1.0;
float TEMP_THRESHOLD = 0.2;
float MAG_THRESHOLD = 1.5;
float R_THRESHOLD = 1.0;
float G_THRESHOLD = 1.0;
float B_THRESHOLD = 1.0;
float CLEAR_THRESHOLD = 1.0;


float prev_avg_rh, prev_avg_temp;
float prev_avg_mag_x, prev_avg_mag_y, prev_avg_mag_z;
float prev_avg_r, prev_avg_g, prev_avg_b, prev_avg_clear;
float curr_avg_rh, curr_avg_temp;
float curr_avg_mag_x, curr_avg_mag_y, curr_avg_mag_z;
float curr_avg_r, curr_avg_g, curr_avg_b, curr_avg_clear;


float magnitude(float x, float y, float z) {
    float mag = 0.0;
    mag += x*x;
    mag += y*y;
    mag += z*z;

    return sqrtf(mag);
}


void report_raw(float rh, float temp, float mag, int r, int g, int b, int clear) {
    Serial.print("raw");
    Serial.print(",");
    Serial.print("rh=");
    Serial.print(rh, 2);
    Serial.print(",");
    Serial.print("temp=");
    Serial.print(temp, 2);
    Serial.print(",");
    Serial.print("mag=");
    Serial.print(mag, 3);
    Serial.print(",");
    Serial.print("r=");
    Serial.print(r);
    Serial.print(",");
    Serial.print("g=");
    Serial.print(g);
    Serial.print(",");
    Serial.print("b=");
    Serial.print(b);
    Serial.print(",");
    Serial.print("clear=");
    Serial.println(clear);
    return;
}

void report_flags(int humid_jump, int temp_rise, int mag_shift, int light_or_color_change) {
    Serial.print("flags");
    Serial.print(",");
    Serial.print("humid_jump=");
    Serial.print(humid_jump);
    Serial.print(",");
    Serial.print("temp_rise=");
    Serial.print(temp_rise);
    Serial.print(",");
    Serial.print("mag_shift=");
    Serial.print(mag_shift);
    Serial.print(",");
    Serial.print("light_or_color_change=");
    Serial.println(light_or_color_change);
    return;
}


int process_rh_temp(unsigned long currentTime, float raw_rh, float raw_temp, float raw_mag, int raw_r, int raw_g, int raw_b, int raw_clear){

    int humid_jump = 0;
    int temp_rise = 0;

    prev_avg_rh = curr_avg_rh;
    prev_avg_temp = curr_avg_temp;

    curr_avg_rh = (ALPHA * raw_rh) + ((1.0 - ALPHA) * curr_avg_rh);
    curr_avg_temp = (ALPHA * raw_temp) + ((1.0 - ALPHA) * curr_avg_temp);

    if (curr_avg_rh - prev_avg_rh > RH_THRESHOLD) {
        humid_jump = 1;
    }

    if (curr_avg_temp - prev_avg_temp > TEMP_THRESHOLD) {
        temp_rise = 1;
    }


    if ((humid_jump == 1) || (temp_rise == 1)) {
        if (last_breathWarmAir_TriggerTime == 0 || (currentTime - last_breathWarmAir_TriggerTime >= eventTriggerLockOutDuration)) {
            report_raw(raw_rh, raw_temp, raw_mag, raw_r, raw_g, raw_b, raw_clear);
            report_flags(humid_jump, temp_rise, 0, 0);
            Serial.println("event,BREATH_OR_WARM_AIR_EVENT");
            last_breathWarmAir_TriggerTime = currentTime;
            return 1;
        } else {
            return 0;
        }
    }
    return 0;
}

int process_mag(unsigned long currentTime, float raw_rh, float raw_temp, float raw_mag, int raw_r, int raw_g, int raw_b, int raw_clear, float raw_mag_x, float raw_mag_y, float raw_mag_z){

    float prev_avg_mag, curr_avg_mag;

    prev_avg_mag = magnitude(prev_avg_mag_x, prev_avg_mag_y, prev_avg_mag_z);

    prev_avg_mag_x = curr_avg_mag_x;
    prev_avg_mag_y = curr_avg_mag_y;
    prev_avg_mag_z = curr_avg_mag_z;

    curr_avg_mag_x = (ALPHA * raw_mag_x) + ((1.0 - ALPHA) * curr_avg_mag_x);
    curr_avg_mag_y = (ALPHA * raw_mag_y) + ((1.0 - ALPHA) * curr_avg_mag_y);
    curr_avg_mag_z = (ALPHA * raw_mag_z) + ((1.0 - ALPHA) * curr_avg_mag_z);

    curr_avg_mag = magnitude(curr_avg_mag_x, curr_avg_mag_y, curr_avg_mag_z);

    if (fabs(curr_avg_mag - prev_avg_mag) > MAG_THRESHOLD) {
        if (last_magneticDisturbance_TriggerTime == 0 || (currentTime - last_magneticDisturbance_TriggerTime >= eventTriggerLockOutDuration)) {
            report_raw(raw_rh, raw_temp, raw_mag, raw_r, raw_g, raw_b, raw_clear);
            report_flags(0, 0, 1, 0);
            Serial.println("event,MAGNETIC_DISTURBANCE_EVENT");
            last_magneticDisturbance_TriggerTime = currentTime;
            return 1;
        } else {
            return 0;
        }
    }

    return 0;
}

int process_rgb_clear(unsigned long currentTime, float raw_rh, float raw_temp, float raw_mag, int raw_r, int raw_g, int raw_b, int raw_clear){
    prev_avg_r = curr_avg_r;
    prev_avg_g = curr_avg_g;
    prev_avg_b = curr_avg_b;
    prev_avg_clear = curr_avg_clear;

    curr_avg_r = (ALPHA * raw_r) + ((1.0 - ALPHA) * curr_avg_r);
    curr_avg_g = (ALPHA * raw_g) + ((1.0 - ALPHA) * curr_avg_g);
    curr_avg_b = (ALPHA * raw_b) + ((1.0 - ALPHA) * curr_avg_b);
    curr_avg_clear = (ALPHA * raw_clear) + ((1.0 - ALPHA) * curr_avg_clear);

    if ((fabs(curr_avg_r - prev_avg_r) > R_THRESHOLD)
      || (fabs(curr_avg_g - prev_avg_g) > G_THRESHOLD)
      || (fabs(curr_avg_b - prev_avg_b) > B_THRESHOLD)
      || (fabs(curr_avg_clear - prev_avg_clear) > CLEAR_THRESHOLD)
    ) {
        if (last_lightColorChange_TriggerTime == 0 || (currentTime - last_lightColorChange_TriggerTime >= eventTriggerLockOutDuration)) {
            report_raw(raw_rh, raw_temp, raw_mag, raw_r, raw_g, raw_b, raw_clear);
            report_flags(0, 0, 0, 1);
            Serial.println("event,LIGHT_OR_COLOR_CHANGE_EVENT");
            last_lightColorChange_TriggerTime = currentTime;
            return 1;
        } else {
            return 0;
        }
    }

    return 0;
}

void evaluate(float raw_rh, float raw_temp, float raw_mag_x, float raw_mag_y, float raw_mag_z, int raw_r, int raw_g, int raw_b, int raw_clear){
    float raw_mag;
    int breath_or_warm_air = 0;
    int magnetic_disturbance_event = 0;
    int light_or_color_change_event = 0;

    raw_mag = magnitude(raw_mag_x, raw_mag_y, raw_mag_z);

    unsigned long currentTime = millis();

    breath_or_warm_air = process_rh_temp(currentTime, raw_rh, raw_temp, raw_mag, raw_r, raw_g, raw_b, raw_clear);
    magnetic_disturbance_event = process_mag(currentTime, raw_rh, raw_temp, raw_mag, raw_r, raw_g, raw_b, raw_clear, raw_mag_x, raw_mag_y, raw_mag_z);
    light_or_color_change_event = process_rgb_clear(currentTime, raw_rh, raw_temp, raw_mag, raw_r, raw_g, raw_b, raw_clear);

    if((breath_or_warm_air == 0) && (magnetic_disturbance_event == 0) && (light_or_color_change_event == 0)) {
        if (last_baseline_TriggerTime == 0 || (currentTime - last_baseline_TriggerTime >= baselineTriggerLockOutDuration)) {
            report_raw(raw_rh, raw_temp, raw_mag, raw_r, raw_g, raw_b, raw_clear);
            report_flags(0, 0, 0, 0);
            Serial.println("event,BASELINE_NORMAL");
            last_baseline_TriggerTime = currentTime;
        }
    }
}


void setup() {
    Serial.begin(115200);
    delay(1500);

    if(!HS300x.begin()) {
        Serial.println("Failed to initialize humidity/temperature sensor.");
        while(1);
    }

    if(!IMU.begin()) {
        Serial.println("Failed to initialize IMU.");
        while (1);
    }

    if(!APDS.begin()) {
        Serial.println("Failed to initialize APDS9960 sensor.");
        while(1);
    }

    Serial.println("Indoor Environment Change Detector started");

    delay(1000);

}

void loop() {
    float rh, temp;
    float mag_x, mag_y, mag_z;
    int   r, g, b, clear;

    int   mag_ready = 0;
    int   rgb_ready = 0;


    if (IMU.magneticFieldAvailable()) {
        mag_ready = 1;
    }

    if (APDS.colorAvailable()) {
        rgb_ready = 1;
    }

    if ((mag_ready == 1) && (rgb_ready == 1)) {
        temp = HS300x.readTemperature();
        rh = HS300x.readHumidity();
        IMU.readMagneticField(mag_x, mag_y, mag_z);
        APDS.readColor(r, g, b, clear);

        if (!isInitialized) {
            prev_avg_rh = rh;
            prev_avg_temp = temp;
            curr_avg_rh = rh;
            curr_avg_temp = temp;

            prev_avg_mag_x = mag_x;
            prev_avg_mag_y = mag_y;
            prev_avg_mag_z = mag_z;
            curr_avg_mag_x = mag_x;
            curr_avg_mag_y = mag_y;
            curr_avg_mag_z = mag_z;

            prev_avg_r = (float) r;
            prev_avg_g = (float) g;
            prev_avg_b = (float) b;
            prev_avg_clear = (float) clear;
            curr_avg_r = (float) r;
            curr_avg_g = (float) g;
            curr_avg_b = (float) b;
            curr_avg_clear = (float) clear;

            isInitialized = true;
        }

        evaluate(rh, temp, mag_x, mag_y, mag_z, r, g, b, clear);

    }

    delay(100);
}
