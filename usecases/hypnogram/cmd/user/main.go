package main

import (
	"SPADE/usecases/models"
	"SPADE/utils"
	"bufio"
	"fmt"
	"os"
	"strconv"
)

func _ReadHypnogramFile(path string, normVal int) []int {
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Error opening file %s: %v\n", path, err)
		return nil
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	data := make([]int, 0)

	for scanner.Scan() {
		item, err := strconv.Atoi(scanner.Text())
		if err != nil {
			fmt.Printf("Error parsing integer from file %s: %v\n", path, err)
			continue
		}
		data = append(data, item+normVal)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Error scanning file %s: %v\n", path, err)
		return nil
	}

	return data
}

// InitUser Upload hypnogram at filePath with
func InitUser(serverAddr string, id int, filePath string) {
	config := models.NewConfig(DbName, TbName, NumUsers, MaxVecSize, PaddingItem, TimeOut, MaxMsgSize)
	hypnogram := _ReadHypnogramFile(filePath, 1)
	data := utils.AddPadding(PaddingItem, MaxVecSize, hypnogram)
	models.StartUser(serverAddr, config, id, data, int32(len(hypnogram)))
}

func main() {
	if len(os.Args) < 3 {
		panic("Usage: ./user <server_address> <id> <path_to_file>")
	}
	serverAddr := os.Args[1]
	id, err := strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Println("Error converting id to integer", err)
		panic(err)
	}
	filePath := os.Args[3]
	InitUser(serverAddr, id, filePath)
}
