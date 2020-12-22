.PHONY: all
all: task

task: task.cpp
	g++ task.cpp -o task -lOpenCL -Wno-deprecated-declarations -lgomp -fopenmp -std=c++20

clean:
	rm -vf task
