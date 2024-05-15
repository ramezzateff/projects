#include "main.h"

/*
 * main: AN Entry Point.
 * @choice: int var store the number you choose from the menu
 */
int main()
{
    int choice;
    loadTasksFromFile();

    cout << "\n\t\t| WELCOME:              |\n";
    while (true)
    {
        cout << "\t\t|   1. Add Task         |\n";
        cout << "\t\t|   2. View Tasks       |\n";
        cout << "\t\t|   3. Delete Task      |\n";
        cout << "\t\t|   4. Exit             |\n";
        cout << "\t\t|   Enter your choice: ";
        cin >> choice; //read user choice.

        if(cin.fail()) //check if input is failed
            {
                cin.clear(); // clear error flag
                cin.ignore(numeric_limits<streamsize>::max(), '\n'); // discard invalid input
                cout << "Invalid choice. Please enter an integer.\n";
                continue; // resstart the loop
            }

        switch (choice)
        {
            case 1: addTask(); break;
            case 2: viewTasks(); break;
            case 3: deleteTask(); break;
            case 4: return 0;
            default: cout << "Invalid choice.\n";
        }
    }

    return (0);
}
