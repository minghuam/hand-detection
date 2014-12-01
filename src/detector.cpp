#include "detector.hpp"
#include "tinylog.hpp"

std::string Detector::key() const{
	return _key;
}

double Detector::train_cost() const{
	return _train_cost;
}

double Detector::detect_cost() const{
	return _detect_cost;
}

void Detector::print() const{
	LOGF("DETECTOR: key: %s, train_cost: %.8f, detect_cost: %.8f",\
		_key.c_str(), _train_cost, _detect_cost);
}