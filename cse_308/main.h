#ifndef MAIN_H
#define MAIN_H
/*
 * @file main.h
 * Header file containing function prototypes
 *	   for the To-Do List application.
 */
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <limits>

using namespace std;

void loadTasksFromFile(void);
void saveTasksToFile(void);
void addTask(void);
void viewTasks(void);
void deleteTask(void);


#endif /* MAIN_H */
