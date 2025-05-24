#include <Mouse.h>

void setup() {
  Serial.begin(9600);
  Mouse.begin();
}

void loop() {
  
  if (Serial.available()) {

    String input = Serial.readStringUntil('\n');

    if (input.startsWith("MOVE")) {

      int dx = 0, dy = 0;
      sscanf(input.c_str(), "MOVE %d %d", &dx, &dy);
      Mouse.move(dx, dy);

    } 
    else if (input == "FIRE") {

      Mouse.press(MOUSE_LEFT);
      Mouse.release(MOUSE_LEFT);

    }
  }
}
