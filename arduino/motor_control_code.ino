#include <Adafruit_MotorShield.h>

// ==========================================
// CONFIGURATION
// ==========================================
const int EXIT_VALUE = 256;         // Command to coast motors
const int BAUD_RATE = 115200;
const int BRAKE_DURATION_POS = 10;  // ms to reverse thrust when stopping from positive
const int BRAKE_DURATION_NEG = 25;  // ms to reverse thrust when stopping from negative
const int FULL_SPEED = 255;

// ==========================================
// OBJECTS
// ==========================================
Adafruit_MotorShield AFMS = Adafruit_MotorShield();
Adafruit_DCMotor *turretMotor = AFMS.getMotor(1);

// Global state for parsing
int currentVal = 0;
bool isNegative = false;
bool isPacketInProgress = false;

// Global state for motor logic
int previousSpeed = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  
  if (!AFMS.begin()) {
    // If shield fails, just halt
    while (1);
  }
  
  turretMotor->setSpeed(0);
  turretMotor->run(RELEASE);
}

void loop() {
  // Fast reading of the serial buffer
  while (Serial.available() > 0) {
    char c = Serial.read();

    // 1. Check for Negative Sign
    if (c == '-') {
      isNegative = true;
      isPacketInProgress = true;
      currentVal = 0; // Reset
    }
    // 2. Check for Digits (ASCII Math)
    else if (c >= '0' && c <= '9') {
      isPacketInProgress = true;
      // Standard ASCII math: shift current value left, add new digit
      // e.g., have 1, receive 2 -> 1*10 + 2 = 12
      currentVal = (currentVal * 10) + (c - '0');
    }
    // 3. Check for Terminator (Newline sent by Python)
    else if (c == '\n') {
      if (isPacketInProgress) {
        if (isNegative) {
          currentVal = -currentVal;
        }
        
        // Execute command immediately
        handleMotorControl(currentVal);

        // Reset for next packet
        currentVal = 0;
        isNegative = false;
        isPacketInProgress = false;
      }
    }
  }
}

void handleMotorControl(int speed) {
  
  // CASE 1: Exit/Coast Command
  if (speed == EXIT_VALUE) {
    turretMotor->run(RELEASE);
    previousSpeed = 0;
    return;
  }

  // CASE 2: Stop Command (Active Braking)
  if (speed == 0) {
    // Only brake if we were previously moving
    if (previousSpeed != 0) {
        applyActiveBraking();
    }
    turretMotor->setSpeed(0);
    turretMotor->run(RELEASE);
    previousSpeed = 0;
  }
  // CASE 3: Movement Command
  else {
    if (speed > 0) {
      turretMotor->run(FORWARD);
      turretMotor->setSpeed(speed);
    } else {
      turretMotor->run(BACKWARD);
      turretMotor->setSpeed(abs(speed));
    }
    previousSpeed = speed;
  }
}

// Applies a short burst of reverse polarity to stop momentum quickly
void applyActiveBraking() {
  if (previousSpeed < 0) {
    // Was moving backward, pulse forward
    turretMotor->run(FORWARD);
    turretMotor->setSpeed(FULL_SPEED);
    delay(BRAKE_DURATION_NEG);
  } 
  else if (previousSpeed > 0) {
    // Was moving forward, pulse backward
    turretMotor->run(BACKWARD);
    turretMotor->setSpeed(FULL_SPEED);
    delay(BRAKE_DURATION_POS);
  }
}