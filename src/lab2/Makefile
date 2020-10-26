all: task

task: task.cpp
	g++ task.cpp -o task -lOpenCL -Wno-deprecated-declarations -lgomp -fopenmp

clean:
	rm -vf task
