package main

import (
	"SPADE/usecases/models"
	"SPADE/utils"
	"fmt"
	"os"
	"strconv"
)

// InitUser Upload hypnogram at filePath with id
func InitUser(serverAddr string, id int, filePath string) {
	config := models.NewConfig(DbName, TbName, NumUsers, MaxVecSize, PaddingItem, TimeOut, MaxMsgSize)
	hypnogram := utils.ReadHypnogramFile(filePath)
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
