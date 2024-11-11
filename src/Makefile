# Compiler
CC = gcc

# Compiler flags
CFLAGS = -I./include -Wall -Wextra -std=c17

# Linker flags
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -lopenblas -lm

# Source files
SRCS = main.c data_loader.c network.c utils.c trainer.c

# Object files
OBJS = $(SRCS:.c=.o)

# Executable name
EXEC = main.bin

# Default target
all: $(EXEC)

# Link object files to create the executable
$(EXEC): $(OBJS)
	$(CC) $(OBJS) -o $(EXEC) $(LDFLAGS)

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(EXEC)

# Phony targets
.PHONY: all clean
