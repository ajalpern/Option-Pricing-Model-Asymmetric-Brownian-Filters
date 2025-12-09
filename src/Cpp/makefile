# Define the base compiler and flags
CXX         = /usr/local/bin/mpicxx
BASE_CXXFLAGS = -Wall -std=c++17
LDFLAGS = -lboost_json -lboost_system

SRCS        = MCFit.cpp AzzModel.cpp util.cpp BinaryOptionTreeFuncs.cpp
OBJS        = $(SRCS:.cpp=.o)

# Default target
all: fit_model

# --- Production Build (Default) ---

PROD_CXXFLAGS = $(BASE_CXXFLAGS) -O2
PROD_EXEC     = fit_model

fit_model: $(OBJS)
	$(CXX) $(OBJS) -o $(PROD_EXEC) $(PROD_CXXFLAGS) $(LDFLAGS)

# --- Debug Build Target ---

# Define flags and output name for the DEBUG build
DEBUG_CXXFLAGS = $(BASE_CXXFLAGS) -DDEBUG -g
DEBUG_EXEC     = fit_model_debug

# The DEBUG target explicitly needs to build the DEBUG executable
debug: $(DEBUG_EXEC)

$(DEBUG_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $(DEBUG_EXEC) $(DEBUG_CXXFLAGS) $(LDFLAGS)

# --- Compilation Rules with Dynamic Flags and Dependencies ---

# Define a variable for the dynamic flags selection logic:
# It selects DEBUG_CXXFLAGS if "DEBUG" is in the command goals, otherwise PROD_CXXFLAGS.
DYNAMIC_CXXFLAGS = $(if $(filter debug,$(MAKECMDGOALS)),$(DEBUG_CXXFLAGS),$(PROD_CXXFLAGS))


MCFit.o: MCFit.cpp AzzModel.h util.h
	$(CXX) $(DYNAMIC_CXXFLAGS) -c $< -o $@


AzzModel.o: AzzModel.cpp util.h BinaryOptionTreeFuncs.h
	$(CXX) $(DYNAMIC_CXXFLAGS) -c $< -o $@


util.o: util.cpp util.h
	$(CXX) $(DYNAMIC_CXXFLAGS) -c $< -o $@


BinaryOptionTreeFuncs.o: BinaryOptionTreeFuncs.cpp BinaryOptionTreeFuncs.h
	$(CXX) $(DYNAMIC_CXXFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(DYNAMIC_CXXFLAGS) -c $< -o $@

# Phony targets
.PHONY: all debug clean

clean:
	rm $(OBJS) $(PROD_EXEC) $(DEBUG_EXEC)
