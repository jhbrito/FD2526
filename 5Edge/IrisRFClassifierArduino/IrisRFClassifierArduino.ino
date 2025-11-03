#include "IrisClassifier.h"

// IrisClassifier.h creates a irisClassifier object
// that you can use to classify a feature vector
// no setup is required

void setup() {
    Serial.begin(115200);
}

void loop() {
    // replace with your actual feature vector
    float input0[4] = {5.1, 3.5, 1.4, 0.2};
    float input1[4] = {7,3.2,4.7,1.4};
    float input2[4] = {6.3,3.3,6,2.5};


    Serial.print("Prediction 0: ");
    
    int i0 = micros();
    int c0 = irisClassifier.predict(input0);
    int dt0 = micros() - i0;
    
    Serial.print(c0);
    Serial.print(" - ");
    Serial.print(dt0);
    Serial.println(" us");

    Serial.print("Prediction 1: ");
    
    int i1 = micros();
    int c1 = irisClassifier.predict(input1);
    int dt1 = micros() - i1;
    
    Serial.print(c1);
    Serial.print(" - ");
    Serial.print(dt1);
    Serial.println(" us");


Serial.print("Prediction 2: ");
    
    int i2 = micros();
    int c2 = irisClassifier.predict(input2);
    int dt2 = micros() - i2;
    
    Serial.print(c2);
    Serial.print(" - ");
    Serial.print(dt2);
    Serial.println(" us");
    delay(1000);
}