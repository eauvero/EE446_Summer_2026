#include <PDM.h>
#include <Arduino_APDS9960.h>
#include <Arduino_BMI270_BMM150.h>
#include <math.h>
#include <string.h>

int SOUND_THRESHOLD = 50;
int DARK_THRESHOLD = 20;
float MOVING_THRESHOLD = 0.01;
int NEAR_THRESHOLD = 100;

float baselineMag = 0.0;

short sampleBuffer[256];
volatile int samplesRead = 0;

int evaluate_sound(int mic) {

    if (mic > SOUND_THRESHOLD) {
        return 1;
    }

    return 0;
}

int evaluate_dark(int clear) {

    if (clear < DARK_THRESHOLD) {
        return 1;
    }

    return 0;
}

int evaluate_moving(float motion) {

    if (abs(motion - baselineMag)  > MOVING_THRESHOLD) {
        return 1;
    }

    return 0;
}

int evaluate_near(int prox) {
    if (prox < NEAR_THRESHOLD) {
        return 1;
    }

    return 0;
}



void report_raw(int mic, int clear, float motion, int prox) {
    Serial.print("raw");
    Serial.print(",");
    Serial.print("mic=");
    Serial.print(mic);
    Serial.print(",");
    Serial.print("clear=");
    Serial.print(clear);
    Serial.print(",");
    Serial.print("motion=");
    Serial.print(motion, 3);
    Serial.print(",");
    Serial.print("prox=");
    Serial.println(prox);
    return;
}

void report_flags(int sound, int dark, int moving, int near) {
    Serial.print("flags");
    Serial.print(",");
    Serial.print("sound=");
    Serial.print(sound);
    Serial.print(",");
    Serial.print("dark=");
    Serial.print(dark);
    Serial.print(",");
    Serial.print("moving=");
    Serial.print(moving);
    Serial.print(",");
    Serial.print("near=");
    Serial.println(near);
    return;
}


void evaluate(int mic, int clear, float motion, int prox){
    int sound = evaluate_sound(mic);
    int dark = evaluate_dark(clear);
    int moving = evaluate_moving(motion);
    int near = evaluate_near(prox);

    int state_id = (sound << 3) | (dark << 2) | (moving << 1) | near;
    const char* final_state = "";

    switch(state_id) {
        case 0: final_state = "QUIET_BRIGHT_STEADY_FAR"; break;   //0000
        case 8: final_state = "NOISY_BRIGHT_STEADY_FAR"; break;   //1000
        case 5: final_state = "QUIET_DARK_STEADY_NEAR"; break;     //0101
        case 11: final_state = "NOISY_BRIGHT_MOVING_NEAR"; break; //1011
    }

    if (strlen(final_state) > 0) {
        report_raw(mic, clear, motion, prox);
        report_flags(sound, dark, moving, near);
        Serial.print("state,");
        Serial.println(final_state);
    }

    return;
}


void onPDMdata() {
    int bytesAvailable = PDM.available();
    PDM.read(sampleBuffer, bytesAvailable);
    samplesRead = bytesAvailable / 2;
}


float mag_accel(float ax, float ay, float az){
    float mag = 0.0;
    mag += ax*ax;
    mag += ay*ay;
    mag += az*az;
    return sqrtf(mag);
}

void setup() {
    Serial.begin(115200);
    delay(1500);
    bool baseline_taken = false;
    float ax, ay, az;

    PDM.onReceive(onPDMdata);

    if(!PDM.begin(1, 16000)) {
        Serial.println("Failed to start PDM microphone.");
        while (1);
    }

    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU.");
        while (1);
    }

    if(!APDS.begin()) {
        Serial.println("Failed to initialize APDS9960 sensor.");
        while(1);
    }

    

    while(!baseline_taken){
        delay(1000);
        if (IMU.accelerationAvailable()) {
            IMU.readAcceleration(ax, ay, az);
            baselineMag = sqrtf(ax*ax + ay*ay + az*az);
            baseline_taken = true;
            Serial.print("Baseline acceleration measured");
        }
    }

    Serial.println("Smart Workspace Situation Classifier started");
}

void loop() {
    int mic;
    int r, g, b, clear;
    float ax, ay, az, motion;
    int prox;

    int rgb_ready = 0;
    int accel_ready = 0;
    int prox_ready = 0;

    if (APDS.colorAvailable()) {
        rgb_ready = 1;
    }

    if (IMU.accelerationAvailable()) {
        accel_ready = 1;
    }

    if (APDS.proximityAvailable()) {
        prox_ready = 1;
    }

    if ((samplesRead) && (rgb_ready == 1) && (accel_ready == 1) && (prox_ready == 1)) {
        long sum = 0;
        for (int i = 0; i < samplesRead; i++) {
            sum += abs(sampleBuffer[i]);
        }
        mic = sum / samplesRead;
        samplesRead = 0;

        APDS.readColor(r, g, b, clear);
        IMU.readAcceleration(ax, ay, az);
        motion = mag_accel(ax, ay, az);
        prox = APDS.readProximity();

        evaluate(mic, clear, motion, prox);
    }
 
    delay(1000);
}
