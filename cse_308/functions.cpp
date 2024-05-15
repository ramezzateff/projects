#include "main.h"

std::vector<std::string> taskNames;
std::vector<std::string> taskDescriptions;
/*
 * load Tasks From File: Load tasks from a file into memory.
 */
void loadTasksFromFile()
{
    ifstream readFile("TO_DO_LIST.txt"); // open file for reading
    if (readFile.is_open()) // check if file is open
    {
        string line;
        while (getline(readFile, line)) // read each line from the file
            {
                if (line.find("Name: ") != string::npos) // Check if the line contains "Name: "
                {
                    taskNames.push_back(line.substr(6)); // Extract task name and add it to taskname vector
                } //check if line cotains description
                else if (line.find("Description: ") != string::npos)
                {
                        taskDescriptions.push_back(line.substr(12));
                        // Extract task description and add it to taskdescription vector
                }
            }
    readFile.close(); // close the file after reading
    }
}

/*
 * save tasks to file: save task to file txt.
 * Return: VOID.
 */
void saveTasksToFile()
{
    ofstream writeFile("TO_DO_LIST.txt"); //open file for writing
    if (!writeFile.is_open()) // check if the file is open
    {
        cout << "Unable to open file.\n"; // if non enable opening, display an error message
        return;
    }

    for (size_t i = 0; i < taskNames.size(); ++i) //  // Write task information to the file
    	{
            writeFile << "Task[ " << i + 1 << " ]:\n"; // Write Task
            writeFile << "Name: " << taskNames[i] << "\n"; // Write Name
            writeFile << "Description: " << taskDescriptions[i] << "\n\n"; // Write Description
        }

    writeFile.close(); // close the file
}

/*
 * add task: add new task to the to do list
 * @name: string variable store the name in taskname vector
 * @description: string variable store the desc in taskdescription vector
 * Return: VOID.
 */
void addTask()
{
    string name, description;

    cout << "Enter task name: ";
    getline(cin >> ws, name); // read task name from user and ignoring leading whitespace.
    cout << "Enter task description: ";
    getline(cin >> ws, description); // read task desc from user and ignore leading whitespace

    taskNames.push_back(name); //put the name of task in vector taskname by using push_back
    taskDescriptions.push_back(description);// put the name of task in vector taskname by using push_back

    saveTasksToFile(); // save to txt file

    cout << "Task added successfully.\n";
}

/*
 * view task: View all tasks in the To-Do List.
 * Return: void.
 */
void viewTasks()
{
    if (taskNames.empty()) // check if no tasks availble
    {
            cout << "No tasks available.\n";
            return;
        }
    // Iterate through all tasks and display their details
    for (size_t i = 0; i < taskNames.size(); ++i){
            cout << "Task[ " << i + 1 << " ]:\n"; // display task number
            cout << "Name: " << taskNames[i] << "\n"; // display name of task
            cout << "Description: " << taskDescriptions[i] << "\n\n";
            //display the task desc.
        }
}
/*
 * delete task: delete a task from the to do list
 * @taskNumber: int variable that has a task number to delete.
 * Return: VOID.
 */
void deleteTask()
{
    int taskNumber;

    cout << "Enter the task number to delete: ";
    cin >> taskNumber; // read from user task no to delete

    if(cin.fail()) //check if input is failed
    {
     	cin.clear(); // clear error flag
        cin.ignore(numeric_limits<streamsize>::max(), '\n'); // discard invalid input
        cout << "Invalid choice. Please enter an integer.\n";
       // continue; // resstart the loop
            }
    // check if task number is valid.
    if (taskNumber > 0 && taskNumber <= taskNames.size()){
            taskNames.erase(taskNames.begin() + taskNumber - 1);
            taskDescriptions.erase(taskDescriptions.begin() + taskNumber - 1);
	// Erase the task name and description at the specified index

        saveTasksToFile(); // save the upload task to file

        cout << "Task deleted successfully.\n";
        }
    else {
            cout << "Invalid task number.\n";
            
    }
}
