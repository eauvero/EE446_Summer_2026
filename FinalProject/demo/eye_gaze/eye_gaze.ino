#include <TinyMLShield.h>
#include <TensorFlowLite.h>
 
// Using the memory-efficient mutable resolver
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
 
#include "eye_gaze_lsr_mobilenetv2_64px_alpha005_ptq_model.h"
 
// ==========================================
// Serial message framing (shared with the Python listener)
// ==========================================
const byte SYNC1 = 0xAA;
const byte SYNC2 = 0x55;
const byte TYPE_IMAGE  = 0x01;
const byte TYPE_STRING = 0x02;
    
// ==========================================
// ML Variables & Configuration
// ==========================================
// QQVGA Grayscale image buffer
byte image[160 * 120];
int bytesPerFrame;
 
bool prevButtonState = false;
 
// 170KB is exactly enough for the model
constexpr int kTensorArenaSize = 150 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
 
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
 
const char* class_labels[] = {"left", "right", "straight"};
 
 
// Helper Functions
int quantizeToInt8(float value, TfLiteQuantizationParams q) {
  int32_t quantized = (int32_t)roundf(value / q.scale) + q.zero_point;
  return constrain(quantized, -128, 127);
}
 
float dequantizeInt8(int8_t value, TfLiteQuantizationParams q) {
  return (value - q.zero_point) * q.scale;
}
 
// Sends the current camera buffer as: SYNC1 SYNC2 TYPE_IMAGE <fixed-size raw bytes>
void sendImage() {
  Serial.write(SYNC1);
  Serial.write(SYNC2);
  Serial.write(TYPE_IMAGE);
  Serial.write(image, bytesPerFrame);
}

// Sends a short text message as: SYNC1 SYNC2 TYPE_STRING <length byte> <raw text bytes>
// Length-prefixing (rather than a terminator character) keeps this binary-safe
// and lets the listener know exactly how many bytes to read. Caps at 255 chars.
void sendString(const String &msg) {
  byte len = (byte) min((unsigned int)msg.length(), 255u);
  Serial.write(SYNC1);
  Serial.write(SYNC2);
  Serial.write(TYPE_STRING);
  Serial.write(len);
  for (byte i = 0; i < len; i++) {
    Serial.write(msg[i]);
  }
}

// Setup
void setup() {
  Serial.begin(115200);
  while (!Serial);
  delay(2500); //to slowdown the chip
 
  initializeShield();
  if (!Camera.begin(QQVGA, GRAYSCALE, 1, OV7675)) {
    sendString("\n CRITICAL ERROR: Camera failed to initialize.");
    while (true);
  }
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
  sendString(" Camera initialized.");
 
  // 2. Initialize TensorFlow Lite
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
 
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    sendString(" CRITICAL ERROR: Model schema mismatch.");
    while (true);
  }
 
  // Load the around 12 specific math operations MobileNetV2 uses
  sendString("Loading targeted AI operations...");
  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
 
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddAdd();
  resolver.AddMean();
  resolver.AddPad();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddRelu();
 
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
 
  sendString("Allocating Tensors...");
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    sendString(" CRITICAL ERROR: Tensor allocation failed.");
    while (true);
  }
 
  input = interpreter->input(0);
  output = interpreter->output(0);
  sendString(" ML Model loaded successfully.");
  sendString("=====================================\n");
  sendString("Ready. Press the shield button to take a snapshot.");
}
 
 
 
 
 
 
 
void loop() {
  bool pressed = readShieldButton();
 
  // Only trigger on the rising edge (button just pressed), so a single
  // press-and-release runs exactly one capture+classification cycle,
  // regardless of how readShieldButton() behaves while the button is held.
  bool triggered = pressed && !prevButtonState;
  prevButtonState = pressed;
 
  if (triggered) {
    sendString("-------------------------------------");
    sendString("[Taking Snapshot...]");
 
    // --- PART 1: CAPTURE AND CHECK IMAGE QUALITY ---
    unsigned long startCapture = millis();
    Camera.readFrame(image);
    unsigned long endCapture = millis();
 
    int totalPixels = Camera.width() * Camera.height();
    long pixelSum = 0;
    for (int i = 0; i < totalPixels; i++) {
      pixelSum += image[i];
    }
    int averageBrightness = pixelSum / totalPixels;
 
    sendString(">>> STEP 1: IMAGE QUALITY");
    sendString("Capture Time:   " + String(endCapture - startCapture) + " ms");
    sendString("Avg Brightness: " + String(averageBrightness) + " / 255");
 
    if (averageBrightness < 10) {
      sendString("WARNING: Image is completely dark! Inference may fail.");
    } else if (averageBrightness > 245) {
      sendString("WARNING: Image is completely washed out/white!");
    }
 
    sendImage();
 
    // --- PART 2: ML CLASSIFICATION ---
    sendString(">>> STEP 2: ML CLASSIFICATION");
 
    const int crop_width = 64;
    const int crop_height = 64;
    const int start_x = (Camera.width() - crop_width) / 2;
    const int start_y = (Camera.height() - crop_height) / 2;
 
    for (int y = 0; y < crop_height; ++y) {
      for (int x = 0; x < crop_width; ++x) {
        byte pixel = image[(start_y + y) * Camera.width() + start_x + x];
        float normalized_pixel = pixel / 255.0f;
        input->data.int8[y * crop_width + x] = (int8_t)quantizeToInt8(normalized_pixel, input->params);
      }
    }
 
    unsigned long startInference = millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long endInference = millis();
 
    if (invoke_status != kTfLiteOk) {
      sendString(" Inference failed.");
      return;
    }
 
    sendString("Inference Time: " + String(endInference - startInference) + " ms");
 
    int best_idx = 0;
    int8_t best_score = output->data.int8[0];
//updateing the state for the classification
    for (int i = 0; i < 3; ++i) {
      int8_t score = output->data.int8[i];
      sendString(" - " + String(class_labels[i]) + ":\t" + String(dequantizeInt8(score, output->params), 4));
 
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }
 
    sendString("WINNING CLASSIFICATION: -> " + String(class_labels[best_idx]) + " <-");
    sendString("-------------------------------------");
 
  }
}