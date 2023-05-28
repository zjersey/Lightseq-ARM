#pragma once
#include <iostream>
#include <string>
#include <unordered_map>

class Profiler {
private:
    std::unordered_map<std::string, double> record_start;
public:
    void set_start(const std::string& name);
    double set_end(const std::string& name);
};