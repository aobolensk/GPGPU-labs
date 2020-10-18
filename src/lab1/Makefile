all: task1 task2 task3

task1: task1.cpp
	g++ task1.cpp -o task1 -lOpenCL

task2: task2.cpp
	g++ task2.cpp -o task2 -lOpenCL -Wno-deprecated-declarations

task3: task3.cpp
	g++ task3.cpp -o task3 -lOpenCL -Wno-deprecated-declarations

clean:
	rm -vf task1 task2 task3
