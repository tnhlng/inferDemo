#include "application.h"
using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    Application app;
    app.init();
    app.runFolder("../face");
    //app.runCamera();
    return 0;

}