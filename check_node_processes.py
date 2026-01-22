import subprocess 

def find_node_processes(): 
    try: 
        # Run the tasklist command 
        result = subprocess.run([ 
            'tasklist', '/FI', 'IMAGENAME eq node.exe' 
        ], capture_output=True, text=True, shell=True) 
        
        if result.returncode == 0: 
            print("Node.js processes:") 
            print(result.stdout) 
        else: 
            print("Error:", result.stderr) 
            
    except Exception as e: 
        print(f"Error running command: {e}") 

# Usage 
find_node_processes()