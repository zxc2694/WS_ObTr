#ifndef OBJECTTRACKERFACTORY_H
#define OBJECTTRACKERFACTORY_H

#include <memory>
#include "Tracking.h"

using namespace std;

class ObjectTrackerFactory
{
public:
	ObjectTrackerFactory();
	~ObjectTrackerFactory();

	//IObjectTracker* create(std::string tracker_type);
	static shared_ptr<IObjectTracker> create(std::string tracker_type);

private:

};

ObjectTrackerFactory::ObjectTrackerFactory()
{
}

ObjectTrackerFactory::~ObjectTrackerFactory()
{
}

shared_ptr<IObjectTracker> ObjectTrackerFactory::create(std::string tracker_name)
{
	IObjectTracker *instance = nullptr;
	if (tracker_name == "MeanShiftTracker")
		instance = new MeanShiftTracker();
	
	//if (tracker_name == "PFTracker")
	//  instance = new PFTracker;
	//
	if (instance != nullptr)
		return std::shared_ptr<IObjectTracker>(instance);
	else
		return nullptr;
}

#endif