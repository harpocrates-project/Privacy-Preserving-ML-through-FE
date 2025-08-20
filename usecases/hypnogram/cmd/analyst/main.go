package main

import (
	"SPADE/usecases/models"
	"encoding/csv"
	"fmt"
	"log"
	"math/big"
	"os"
	"strconv"
	"strings"
)

// InitAnalyst start the SPADE analyst using hypnogram use-case configuration
func InitAnalyst(serverAddr string, userID int64, queryValue int64, resultsFile string) {
	config := models.NewConfig(DbName, TbName, NumUsers, MaxVecSize, PaddingItem, TimeOut, MaxMsgSize)
	// start analyst entity, send a req to server for getting the cipher belongs to
	// the user with userID and decrypt it for the specific query value queryValue
	queryValue, results := models.StartAnalyst(serverAddr, config, userID, queryValue)

	count := 0
	one := big.NewInt(1)
	for _, v := range results {
		if v.Cmp(one) == 0 {
			count++
		}
	}

	file, err := os.Create(resultsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	wr := csv.NewWriter(file)
	defer wr.Flush()

	st := strings.Fields(strings.Trim(fmt.Sprint(results), "[]"))
	wr.Write(st)

	fmt.Printf("Count: %d\n", count)
}

func main() {
	if len(os.Args) < 4 {
		panic("Usage: ./analyst <server_address> <id> <query_value> <out_path>")
	}
	serverAddr := os.Args[1]
	userID, err := strconv.ParseInt(os.Args[2], 10, 64)

	if err != nil {
		log.Println("Error converting user id to 64bit integer", err)
		panic(err)
	}

	queryValue, err := strconv.ParseInt(os.Args[3], 10, 64)
	if err != nil {
		log.Println("Error converting user id to 64bit integer", err)
		panic(err)
	}

	resultsFile := os.Args[4]

	InitAnalyst(serverAddr, userID, queryValue, resultsFile)
}
