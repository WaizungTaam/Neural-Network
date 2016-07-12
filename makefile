# Creation time: 2016-07-04

CC=g++
STD=c++11
SRC_PATH=./src/
TEST_PATH=./test/
VECTOR=$(SRC_PATH)vector.hh $(SRC_PATH)vector.cc
MATRIX=$(SRC_PATH)matrix.hh $(SRC_PATH)matrix.cc
UTILS=$(SRC_PATH)utils.hh $(SRC_PATH)utils.cc
MATH=$(VECTOR) $(MATRIX) $(UTILS)
PERCEPTRON=$(SRC_PATH)perceptron.hh $(SRC_PATH)perceptron.cc
LAYER=$(SRC_PATH)layer.hh $(SRC_PATH)layer.cc
INPUT_LAYER=$(SRC_PATH)input_layer.hh $(SRC_PATH)input_layer.cc
HIDDEN_LAYER=$(SRC_PATH)hidden_layer.hh $(SRC_PATH)hidden_layer.cc
OUTPUT_LAYER=$(SRC_PATH)output_layer.hh $(SRC_PATH)output_layer.cc
LAYERS=$(INPUT_LAYER) $(HIDDEN_LAYER) $(OUTPUT_LAYER)
LINEAR_MODEL=$(SRC_PATH)linear_model.hh $(SRC_PATH)linear_model.cc

all: vector_test.o matrix_test.o utils_test.o perceptron_test.o layer_test.o \
	 input_layer_test.o hidden_layer_test.o output_layer_test.o mlp_test_1.o linear_model_test.o

linear_model_test.o: $(MATH) $(LAYER) $(LAYERS) $(LINEAR_MODEL) $(TEST_PATH)linear_model_test.cc
	$(CC) -std=$(STD) $(MATH) $(LAYER) $(LAYERS) $(LINEAR_MODEL) $(TEST_PATH)linear_model_test.cc -o linear_model_test.o

mlp_test_1.o: $(MATH) $(LAYER) $(LAYERS) $(TEST_PATH)mlp_test_1.cc
	$(CC) -std=$(STD) $(MATH) $(LAYER) $(LAYERS) $(TEST_PATH)mlp_test_1.cc -o mlp_test_1.o

output_layer_test.o: $(MATH) $(LAYER) $(OUTPUT_LAYER) $(TEST_PATH)output_layer_test.cc
	$(CC) -std=$(STD) $(MATH) $(LAYER) $(OUTPUT_LAYER) $(TEST_PATH)output_layer_test.cc -o output_layer_test.o

hidden_layer_test.o: $(MATH) $(LAYER) $(HIDDEN_LAYER) $(TEST_PATH)hidden_layer_test.cc
	$(CC) -std=$(STD) $(MATH) $(LAYER) $(HIDDEN_LAYER) $(TEST_PATH)hidden_layer_test.cc -o hidden_layer_test.o

input_layer_test.o: $(VECTOR) $(MATRIX) $(LAYER) $(INPUT_LAYER) $(TEST_PATH)input_layer_test.cc
	$(CC) -std=$(STD) $(VECTOR) $(MATRIX) $(LAYER) $(INPUT_LAYER) $(TEST_PATH)input_layer_test.cc -o input_layer_test.o

layer_test.o: $(MATH) $(LAYER) $(TEST_PATH)layer_test.cc
	$(CC) -std=$(STD) $(MATH) $(LAYER) $(TEST_PATH)layer_test.cc -o layer_test.o

perceptron_test.o: $(VECTOR) $(MATRIX) $(PERCEPTRON) $(TEST_PATH)perceptron_test.cc
	$(CC) -std=$(STD) $(VECTOR) $(MATRIX) $(PERCEPTRON) $(TEST_PATH)perceptron_test.cc -o perceptron_test.o

utils_test.o: $(VECTOR) $(MATRIX) $(UTILS) $(TEST_PATH)utils_test.cc
	$(CC) -std=$(STD) $(VECTOR) $(MATRIX) $(UTILS) $(TEST_PATH)utils_test.cc -o utils_test.o

matrix_test.o: $(VECTOR) $(MATRIX) $(TEST_PATH)matrix_test.cc
	$(CC) -std=$(STD) $(VECTOR) $(MATRIX) $(TEST_PATH)matrix_test.cc -o matrix_test.o
 
vector_test.o: $(VECTOR) $(TEST_PATH)vector_test.cc
	$(CC) -std=$(STD) $(VECTOR) $(TEST_PATH)vector_test.cc -o vector_test.o

clean:
	rm *.o