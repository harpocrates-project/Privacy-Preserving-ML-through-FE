package main

import (
	"SPADE/usecases/models"
	"fmt"
	"os"
)

// InitServer start the SPADE server using hypnogram use-case configuration
func InitServer(serverAddr string) {
	config := models.NewConfig(DbName, TbName, NumUsers, MaxVecSize, PaddingItem, TimeOut, MaxMsgSize)
	models.StartServer(serverAddr, config)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: ./server <server_address>")
	}
	serverAddr := os.Args[1]
	InitServer(serverAddr)
}
