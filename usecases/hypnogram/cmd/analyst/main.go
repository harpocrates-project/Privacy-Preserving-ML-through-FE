package main

import (
	"SPADE/usecases/models"
	"encoding/csv"
	"fmt"
	"log"
	"math/big"
	"os"
	"strconv"
)

// InitAnalyst start the SPADE analyst using hypnogram use-case configuration
func InitAnalyst(serverAddr string, userID int64, queryValue int64, normVal int64, resultsFile string) {
	config := models.NewConfig(DbName, TbName, NumUsers, MaxVecSize, PaddingItem, TimeOut, MaxMsgSize)
	// start analyst entity, send a req to server for getting the cipher belongs to
	// the user with userID and decrypt it for the specific normalized queryValue
	_, results := models.StartAnalyst(serverAddr, config, userID, queryValue+normVal)

	for i, v := range results {
		if v.Cmp(big.NewInt(1)) != 0 {
			results[i] = big.NewInt(0)
		} else {
			results[i] = big.NewInt(queryValue)
		}
	}

	count := 0
	for _, v := range results {
		if v.Cmp(big.NewInt(queryValue)) == 0 {
			count++
		}
	}

	file, err := os.Create(resultsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	wr := csv.NewWriter(file)
	for _, v := range results {
		if err := wr.Write([]string{v.String()}); err != nil {
			log.Fatal(err)
		}
	}
	wr.Flush()

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

	InitAnalyst(serverAddr, userID, queryValue, 1, resultsFile)
}
