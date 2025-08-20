package models

import (
	"SPADE"
	pb "SPADE/spadeProto"
	"SPADE/utils"
	"context"
	"errors"
	"log"
	"math/big"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"google.golang.org/grpc"
)

// global variable for public parameters
var cur *Curator
var q *big.Int
var g *big.Int

// server Database Handler
var mDBHandler DBHandler

// server for protocol buffer instance
type server struct {
	pb.UnimplementedCuratorServer
}

// Curator we assume that it is trusted
type Curator struct {
	q           *big.Int
	g           *big.Int
	sks         []*big.Int
	pks         []*big.Int
	regKeys     []*big.Int
	ciphertexts [][][]*big.Int
	spade       *SPADE.SPADE
}

// NewCurator creates a new instance of Curator
func NewCurator() *Curator {
	return &Curator{
		q:           nil,
		g:           nil,
		sks:         nil,
		pks:         nil,
		regKeys:     nil,
		ciphertexts: nil,
		spade:       nil,
	}
}

// GetPublicParams called by @User and @Analyst to get access to the public SPADE parameters
func (s *server) GetPublicParams(ctx context.Context, in *pb.PublicParamsReq) (*pb.PublicParamsResp, error) {
	log.Printf("=== Received GetPublicParams req..")

	// print q, g for debug
	utils.PrintBigIntHex("q", cur.q)
	utils.PrintBigIntHex("g", cur.g)

	qBytes := cur.q.Bytes()
	gBytes := cur.g.Bytes()
	mpkBytes := make([][]byte, 0, len(cur.pks)) // Pre-allocate for efficiency
	for _, pk := range cur.pks {
		mpkBytes = append(mpkBytes, pk.Bytes())
	}

	resp := &pb.PublicParamsResp{
		Q:   qBytes,
		G:   gBytes,
		Mpk: mpkBytes,
	}
	log.Printf("=== Send PublicParam Response..")
	utils.PrintMessageSize(resp)

	return resp, nil
}

// UserRequest called by @User to send his/her encrypted data to the server for storage
func (s *server) UserRequest(ctx context.Context, data *pb.UserReq) (*pb.UserResp, error) {
	log.Printf("=== Received User Request..")
	utils.PrintMessageSize(data)

	err := mDBHandler.CreateUsersCipherTable()
	log.Printf("=== Send User Request..")
	if err != nil {
		return &pb.UserResp{Flag: false}, err
	}
	err = mDBHandler.InsertUsersCipher(data)
	if err != nil {
		return &pb.UserResp{Flag: false}, err
	}
	return &pb.UserResp{Flag: true}, nil
}

// Query called by @Analyst to get the corresponding decryption keys for a specific query value
// to be able to partially decrypt a ciphertext vector corresponding to the user id he/she asked
// for it.
func (s *server) Query(ctx context.Context, req *pb.AnalystReq) (*pb.AnalystResp, error) {
	resp := &pb.AnalystResp{
		Dkv:        nil,
		Ciphertext: nil,
		PtSize:     0,
	}
	log.Printf("=== Received Analyst Request..")
	utils.PrintMessageSize(req)
	// need to retrieve the corresponding regKey for the user.id from DB
	row, err := mDBHandler.GetUserReqById(req.Id)
	if err != nil {
		return resp, err
	}

	if row == nil {
		return resp, nil
	}

	// Unmarshal regKey (slice of big.Int)
	regKey := new(big.Int)
	regKey.SetBytes(row.RegKey)

	// derivative the decryption keys for the query value
	dkv := cur.spade.KeyDerivation(int(req.Id), int(req.Value), cur.sks, regKey)

	dkvBytes := make([][]byte, 0, len(dkv)) // Pre-allocate for efficiency
	for _, dk := range dkv {
		dkvBytes = append(dkvBytes, dk.Bytes())
	}

	resp.Dkv = dkvBytes
	resp.Ciphertext = row.Ciphertext
	resp.PtSize = row.PtSize
	log.Printf("=== Send Analyst Response..")
	utils.PrintMessageSize(resp)
	return resp, nil
}

func StartServer(serverAddr string, config *Config) {
	// creating a connection to DB beforehand
	mDBHandler = NewDBHandler(config.DbName, config.TbName)
	// ----------------------------------
	// SPADE calls here :)
	// let's generate the spade public parameters
	q = new(big.Int).Exp(big.NewInt(2), big.NewInt(128), nil)
	q.Add(q, big.NewInt(1))
	g = SPADE.RandomElementInZMod(q)
	// ----------------------------------
	cur = NewCurator()
	cur.q = q
	cur.g = g
	spd := SPADE.NewSpade(q, g, config.MaxVecSize)
	cur.sks, cur.pks = spd.Setup()
	cur.regKeys = make([]*big.Int, config.NumUsers)
	cur.ciphertexts = make([][][]*big.Int, config.NumUsers)
	cur.spade = spd
	// ----------------------------------

	// Register shutdown hook to call DeleteFile upon termination
	defer func() {
		// IMPORTANT: we have to remove database file after we stop the server
		// because technically each time we run the server, we are initiating it
		// using a new set of public parameters, so if you don't remove the previous
		// database, you will faced conflict between the old encrypted data using
		// old set of parameters and the new encrypted data using the new set
		// note: you can keep it as a comment if you want to measure the storage costs
		// log.Printf("Here we go for deleting database..")
		// utils.DeleteFile(config.DbName)
	}()

	// let's create a shutdown context for server
	ctx, cancel := context.WithCancel(context.Background())
	//defer cancel() // Cancel context when main function exits
	var wg sync.WaitGroup
	wg.Add(1)
	go startServerListener(ctx, &wg, config.MaxMsgSize, serverAddr)

	// Listen for termination signals
	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, syscall.SIGINT, syscall.SIGTERM)

	// Wait for termination signal
	<-signalCh

	cancel()

	// wait for shut down process to be finish
	wg.Wait()

	log.Println("Shutting down completed!")
}

// startServerListener creates a goroutine for starting gRPC server and running the server listener
// upon receiving a shut-down signal it handles it in a proper way, because of whatever functionality
// that we might need to call after the server got shut down
func startServerListener(ctx context.Context, wg *sync.WaitGroup, maxMsgSize int, serverAddr string) {
	defer wg.Done()

	// setup server and start listening for clients
	lis, err := net.Listen("tcp", serverAddr)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	opts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(maxMsgSize), // Adjust limit as needed (in bytes)
	}
	s := grpc.NewServer(opts...)
	log.Printf("=== Server starts listening on %s \n", lis.Addr().String())

	pb.RegisterCuratorServer(s, &server{})

	// Serve function in a separate goroutine
	go func() {
		log.Println("=== server goroutine is running!")

		if err := s.Serve(lis); err != nil {
			if !errors.Is(err, grpc.ErrServerStopped) {
				log.Fatalf("failed to serve: %v", err)
			}
		}
	}()

	select {
	case <-ctx.Done():
		log.Println("=== server goroutine is shutting down!")
		s.Stop()
	}
}
