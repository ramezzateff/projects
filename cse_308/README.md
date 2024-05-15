***THIS PROJECT FOR PROGRAMMING COURSE IN UNIVERSITY.***
 - TO DO LIST APPLICATION:

INTRODUCTION:

The To-Do List Application is a simple program designed to manage tasks by allowing users to add, view, and delete tasks. This report provides an overview of the project structure, functionality, and implementation details.

Project files:

- Main.cpp> contains the main function as the entry
- point.
- Func.cpp> contains the functions of to do list program (loadFiles, saveFiles, add view delete Tasks).
Main.h> contains the headers and the prototypes

Features:

- Add Task: user can add a task by write the name of the task and the decription of it and append on txt file
- View Tasks: user can view the list of task by name and description form the file
- Delete Task: users can delete a task by write the number of the task and the file renumbering the tasks


Alogorthims: 

 - Start the program 
 - Check task list from the file 
 - Display a prompt asking the user to enter the number of the task to add view delete task > if the input invalid > display an invalid message
 - if choose 1 > add task by get name and description of the task 
 - if choose 2 > view the tasks from the txt file
 - if choose 3 > ask user for the number of task to delete: Remove the task corresponding to the provided task number from the list of tasks.
 - Save Tasks to File: Write the updated list of tasks to the file "TO_DO_LIST.txt".
 - Print a message confirming that the task has been deleted successfully.
 - end
