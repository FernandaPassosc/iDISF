#==============================================================================
# Paths
#==============================================================================
DEMO_DIR = .

BIN_DIR = $(DEMO_DIR)/bin
EXT_DIR = $(DEMO_DIR)/externals
SRC_DIR = $(DEMO_DIR)/src
LIB_DIR = $(DEMO_DIR)/lib
INCLUDE_DIR = $(DEMO_DIR)/include
OBJ_DIR = $(DEMO_DIR)/obj
PYTHON3_DIR = $(DEMO_DIR)/python3
BUILD_DIR = $(DEMO_DIR)/build

CC = gcc
CXX = g++
CFLAGS = -Wall -fPIC -std=gnu11 -pedantic -Wno-unused-result -O3 -fopenmp
DEMOFLAGS = -Wall -fPIC -pedantic -Wno-unused-result -O3 -fopenmp
CXXFLAGS = -O3 -lstdc++ -fPIC -ffast-math -march=skylake -mfma -Wall -Wno-unused-result
LIBS = -lm

HEADER_INC = -I $(EXT_DIR) -I $(INCLUDE_DIR)
LIB_INC = -L $(LIB_DIR) -lidisf
CINCLUDES = -I $(DEMO_DIR)/arpack-ng/ICB
#CXXINCLUDES = -I $(EXT_DIR)/spectra/include -I /usr/include/eigen3
CXXINCLUDES = -I $(EXT_DIR)/spectra/include -I $(EXT_DIR)/eigen

#==============================================================================
# Rules
#==============================================================================
.PHONY: all c python3 clean lib

all: lib c python3

lib: obj
	$(eval ALL_OBJS := $(wildcard $(OBJ_DIR)/*.o))
	ar csr $(LIB_DIR)/libidisf.a $(ALL_OBJS)


lib2: obj
	$(eval ALL_OBJS := $(wildcard $(OBJ_DIR)/*.o))

obj: \
	$(OBJ_DIR)/Utils.o \
	$(OBJ_DIR)/IntList.o \
	$(OBJ_DIR)/IntLabeledList.o \
	$(OBJ_DIR)/Color.o \
	$(OBJ_DIR)/PrioQueue.o \
	$(OBJ_DIR)/Image.o \
	$(OBJ_DIR)/kmeans.o \
	$(OBJ_DIR)/eigs.o \
	$(OBJ_DIR)/iDISF.o 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(INCLUDE_DIR)/%.h
	$(CC) $(CFLAGS) -c $< -o $@ $(HEADER_INC) $(LIBS)

$(OBJ_DIR)/eigs.o: $(SRC_DIR)/eigs.cpp
	$(CXX) $(CXXFLAGS) -o $(OBJ_DIR)/eigs.o -c $(SRC_DIR)/eigs.cpp $(HEADER_INC) $(CXXINCLUDES) $(LIBS)

c: lib
	$(CXX) $(DEMOFLAGS) $(CXXFLAGS) iDISF_demo.c -o $(BIN_DIR)/iDISF_demo $(HEADER_INC) $(CXXINCLUDES) $(LIB_INC) $(LIBS)

python3: lib
	python3 python3/setup.py install --user --record $(PYTHON3_DIR)/dir_libs.txt
	python3 python3/setup.py clean;
	#python3 python3/setup.py build_ext -b $(PYTHON3_DIR);
	#python3 python3/setup.py clean;
	#python3 python3/setup.py install --record python3/dir_libs.txt

# sudo python3 python3/setup.py install --record files.txt
# sudo python3 -m pip install -U pip numpy

clean:
	rm -rf $(OBJ_DIR)/* ;
	rm -rf $(BIN_DIR)/* ;
	rm -rf $(PYTHON3_DIR)/*.so ;
	rm -r $(BUILD_DIR) ;
	xargs rm -rf < $(PYTHON3_DIR)/dir_libs.txt
	#rm -rf $(OBJ_DIR)/* ;
	#rm -rf $(LIB_DIR)/* ;
	#rm -rf $(BIN_DIR)/* ;
	#rm -rf $(PYTHON3_DIR)/*.so ;
	#rm -rf ./build/* ;
	#rm -f python3/dir_libs.txt

