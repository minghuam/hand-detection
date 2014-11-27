#include "feature.hpp"

Feature::Feature(){
	_dimension = 0;
	_cache = NULL;
}

int Feature::dimension() const{
	return _dimension;
}

void Feature::enable_cache(ImageCache *cache){
	_cache = cache;
}
