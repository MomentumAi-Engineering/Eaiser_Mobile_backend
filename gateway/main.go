package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// Config (Should be env vars)
var (
	PythonCoreURL = "http://localhost:8001"
	ServerPort    = ":8080"
	JWTSecret     = []byte(os.Getenv("JWT_SECRET")) // Ensure this is set!
)

func init() {
	if len(JWTSecret) == 0 {
		fmt.Println("⚠️  WARNING: JWT_SECRET not set. Security is compromised.")
		JWTSecret = []byte("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") // Default from user .env for dev
	} else {
		fmt.Println("🔒 JWT_SECRET loaded.")
	}
}

// --- Rate Limiter ---
type RateLimiter struct {
	mu       sync.Mutex
	visitors map[string]*visitor
}

type visitor struct {
	lastSeen time.Time
	tokens   float64
}

var limiter = &RateLimiter{
	visitors: make(map[string]*visitor),
}

func (rl *RateLimiter) Allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	v, exists := rl.visitors[ip]
	now := time.Now()

	if !exists {
		rl.visitors[ip] = &visitor{lastSeen: now, tokens: 5.0} // Burst 5
		return true
	}

	elapsed := now.Sub(v.lastSeen).Seconds()
	v.tokens += elapsed
	if v.tokens > 5.0 {
		v.tokens = 5.0
	}
	v.lastSeen = now

	if v.tokens >= 1.0 {
		v.tokens -= 1.0
		return true
	}

	return false
}

// --- JWT Validation ---
type Claims struct {
	Sub    string  `json:"sub"` // Email
	ID     string  `json:"id"`
	Role   string  `json:"role"`
	Exp    float64 `json:"exp"`
}

func parseJWT(tokenString string) (*Claims, bool) {
	parts := strings.Split(tokenString, ".")
	if len(parts) != 3 {
		return nil, false
	}

	// 1. Verify Signature
	signingString := parts[0] + "." + parts[1]
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return nil, false
	}

	h := hmac.New(sha256.New, JWTSecret)
	h.Write([]byte(signingString))
	expectedSignature := h.Sum(nil)

	if !hmac.Equal(signature, expectedSignature) {
		return nil, false
	}

	// 2. Decode Payload
	payloadBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, false
	}

	var claims Claims
	if err := json.Unmarshal(payloadBytes, &claims); err != nil {
		return nil, false
	}

	// 3. Verify Expiration
	if time.Now().Unix() > int64(claims.Exp) {
		return nil, false // Expired
	}

	return &claims, true
}

// --- Middleware ---

func rateLimitMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ip := r.RemoteAddr
		if idx := strings.LastIndex(ip, ":"); idx != -1 {
			ip = ip[:idx]
		}

		if !limiter.Allow(ip) {
			http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
			return
		}
		next(w, r)
	}
}

// Context key for Claims? Go stdlib ctx is strict. 
// For now, we just pass verification. Handlers can re-parse if needed or we modify request context.
// Simplified: Auth middleware just gates access.
func authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			http.Error(w, "Unauthorized: No token", http.StatusUnauthorized)
			return
		}

		parts := strings.Split(authHeader, " ")
		if len(parts) != 2 || parts[0] != "Bearer" {
			http.Error(w, "Unauthorized: Invalid format", http.StatusUnauthorized)
			return
		}

		if _, ok := parseJWT(parts[1]); !ok {
			http.Error(w, "Unauthorized: Invalid token", http.StatusUnauthorized)
			return
		}

		next(w, r)
	}
}

// --- Helper for Proxying ---
func proxyRequest(w http.ResponseWriter, r *http.Request, targetURL string) {
	// Create request
	req, err := http.NewRequest(r.Method, targetURL, r.Body)
	if err != nil {
		http.Error(w, "Proxy Error", http.StatusInternalServerError)
		return
	}
	
	// Copy headers
	for name, values := range r.Header {
		for _, value := range values {
			req.Header.Add(name, value)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error contacting Python Core: %v", err)
		http.Error(w, "Service Unavailable", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for name, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

// --- Handlers ---

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status": "gateway_online", "mode": "hybrid_speed"}`))
}

func handleReportStatus(w http.ResponseWriter, r *http.Request) {
	// Extract ID from URL /api/mobile/report-status/{id}
	path := r.URL.Path
	prefix := "/api/mobile/report-status/"
	if !strings.HasPrefix(path, prefix) {
		http.Error(w, "Invalid Path", http.StatusBadRequest)
		return
	}
	id := path[len(prefix):]
	
	targetURL := fmt.Sprintf("%s/internal/report-status/%s", PythonCoreURL, id)
	proxyRequest(w, r, targetURL)
}

func handleHistory(w http.ResponseWriter, r *http.Request) {
	// Extract User ID from Token
	authHeader := r.Header.Get("Authorization")
	parts := strings.Split(authHeader, " ")
	token := parts[1]
	
	claims, ok := parseJWT(token)
	if !ok {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	
	userID := claims.ID
	if userID == "" {
		// Fallback to SUB if ID missing (though unlikely with our auth.py)
		userID = claims.Sub 
	}

	targetURL := fmt.Sprintf("%s/internal/user-history/%s", PythonCoreURL, userID)
	proxyRequest(w, r, targetURL)
}

func handleAuth(w http.ResponseWriter, r *http.Request) {
	// Proxy /api/auth/* -> /internal/auth/*
	path := r.URL.Path
	// Strip /api/auth
	// /api/auth/login -> /internal/auth/login
	// /api/auth/signup-init -> /internal/auth/signup-init
	
	suffix := strings.TrimPrefix(path, "/api/auth")
	targetURL := PythonCoreURL + "/internal/auth" + suffix
	
	proxyRequest(w, r, targetURL)
}

// Main Proxy Handlers (Body Reading is distinct)
func handleLocationVerify(w http.ResponseWriter, r *http.Request) {
	proxyRequest(w, r, PythonCoreURL+"/internal/location-verify")
}

func handleReportIntake(w http.ResponseWriter, r *http.Request) {
	// Ideally we'd inject user_id from token here, but for now strict proxying
	// We assume client sends user_id in form, or we enhance python side to parse token.
	// Since we are "Speed" muscle, we just stream it through.
	// Python side currently validates user_id form field.
	
	// Issue: Go's http.NewRequest with r.Body might fail if body is already read?
	// It's a stream, so we can pass r.Body directly if we haven't read it.
	// Previous implementation read it into memory. Let's optimize to stream if possible,
	// but multipart usually needs boundary rewriting if we change things.
	
	// Simple pass-through:
	proxyRequest(w, r, PythonCoreURL+"/internal/report-intake")
}

func main() {
	// Public Routes
	http.HandleFunc("/api/mobile/health", rateLimitMiddleware(handleHealth))
	http.HandleFunc("/api/auth/", rateLimitMiddleware(handleAuth)) // Login/Signup

	// Protected Routes
	http.HandleFunc("/api/mobile/location-verify", rateLimitMiddleware(authMiddleware(handleLocationVerify)))
	http.HandleFunc("/api/mobile/report", rateLimitMiddleware(authMiddleware(handleReportIntake)))
	http.HandleFunc("/api/mobile/report-status/", rateLimitMiddleware(authMiddleware(handleReportStatus)))
	http.HandleFunc("/api/mobile/history", rateLimitMiddleware(authMiddleware(handleHistory)))

	fmt.Printf("🚀 Go Gateway (Speed) starting on %s\n", ServerPort)
	fmt.Printf("🔗 Proxying to Python Core (Brain) at %s\n", PythonCoreURL)
	
	if err := http.ListenAndServe(ServerPort, nil); err != nil {
		log.Fatal(err)
	}
}
