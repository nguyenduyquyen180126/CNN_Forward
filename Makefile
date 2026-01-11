# Compiler
CXX = g++

# Flags
CXXFLAGS = -std=c++11 -g -Iinclude

# Target executable
TARGET = test.a

# Find all .c files
SRCS = $(wildcard src/*.c) $(wildcard test/*.c)

# Convert .c → .o
OBJS = $(SRCS:.c=.o)

# Default: build & run
all: clean $(TARGET) run

# Link objects to produce executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile .c → .o
%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	./$(TARGET)

# Cleanup
clean:
	rm -f $(OBJS) $(TARGET)
debug:
	gdb $(TARGET)