#include "IrisClassifier.h"

Eloquent::ML::Port::SVM SVM;

void setup() {
    Serial.begin(115200);
}

void loop() {
    // replace with your actual feature vector
    float input[4] = {5.1,3.5,1.4,0.2}; //"Setosa"

    Serial.print("SVM Prediction: ");
    int i = micros();
    int c = SVM.predict(input);
    int dt = micros() - i;
    
    Serial.print(c);
    Serial.print(" (");
    Serial.print(SVM.idxToLabel(c));
    Serial.print(")");
    Serial.print(" - ");
    Serial.print(dt);
    Serial.println(" us");
    delay(1000);

    float input2[4] = {7.0,3.2,4.7,1.4}; //"Versicolor"
    Serial.print("SVM Prediction: ");
    int i2 = micros();
    int c2 = SVM.predict(input2);
    int dt2 = micros() - i2;
    
    Serial.print(c2);
    Serial.print(" (");
    Serial.print(SVM.idxToLabel(c2));
    Serial.print(")");
    Serial.print(" - ");
    Serial.print(dt2);
    Serial.println(" us");
    delay(1000);

    float input3[4] = {6.3,3.3,6,2.5}; //"Virginica"
    Serial.print("SVM Prediction: ");
    int i3 = micros();
    int c3 = SVM.predict(input3);
    int dt3 = micros() - i3;
    
    Serial.print(c3);
    Serial.print(" (");
    Serial.print(SVM.idxToLabel(c3));
    Serial.print(")");
    Serial.print(" - ");
    Serial.print(dt3);
    Serial.println(" us");
    delay(1000);

}