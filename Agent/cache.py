import redis
import json
import time
import hashlib
from config import CACHE_BACKEND, REDIS_CONFIG

class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.use_redis = False
        self.memory_cache = {}
        self.default_ttl = 3600
        self._last_reconnect_attempt = 0
        self._reconnect_interval = 60  # 60秒重试一次
        self.cache_backend = CACHE_BACKEND
        if self.cache_backend == "redis":
            self._try_connect_redis()

    def _try_connect_redis(self):
        """尝试连接 Redis"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_CONFIG['host'],
                port=REDIS_CONFIG['port'],
                db=int(REDIS_CONFIG['db']),
                password=REDIS_CONFIG['password'],
                decode_responses=True
            )
            self.redis_client.ping()
            self.use_redis = True
            print("✓ Redis连接成功")
        except Exception as e:
            print(f"⚠️ Redis连接失败，使用内存缓存: {e}")
            self.use_redis = False

    def _maybe_reconnect(self):
        """在 Redis 操作失败时尝试重新连接"""
        import time
        now = time.time()
        if not self.use_redis and (now - self._last_reconnect_attempt) >= self._reconnect_interval:
            if self.cache_backend != "redis":
                return
            self._last_reconnect_attempt = now
            print("⏳ 尝试重新连接 Redis...")
            self._try_connect_redis()
    
    def get(self, key):
        if self.use_redis:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
                return None
            except Exception as e:
                print(f"Redis get error: {e}")
                self._maybe_reconnect()
                return self._get_memory_cache(key)
        else:
            self._maybe_reconnect()
            return self._get_memory_cache(key)
    
    def set(self, key, value, ttl=None):
        ttl = ttl or self.default_ttl
        if self.use_redis:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
                return True
            except Exception as e:
                print(f"Redis set error: {e}")
                self._maybe_reconnect()
                return self._set_memory_cache(key, value, ttl)
        else:
            self._maybe_reconnect()
            return self._set_memory_cache(key, value, ttl)
    
    def delete(self, key):
        if self.use_redis:
            try:
                self.redis_client.delete(key)
                return True
            except Exception as e:
                print(f"Redis delete error: {e}")
                self._maybe_reconnect()
                return self._delete_memory_cache(key)
        else:
            self._maybe_reconnect()
            return self._delete_memory_cache(key)
    
    def exists(self, key):
        if self.use_redis:
            try:
                return bool(self.redis_client.exists(key))
            except Exception as e:
                print(f"Redis exists error: {e}")
                self._maybe_reconnect()
                return self._exists_memory_cache(key)
        else:
            self._maybe_reconnect()
            return self._exists_memory_cache(key)
    
    # 内存缓存方法
    def _get_memory_cache(self, key):
        if key in self.memory_cache:
            item = self.memory_cache[key]
            if time.time() < item['expiry']:
                return item['value']
            else:
                # 缓存过期
                del self.memory_cache[key]
        return None
    
    def _set_memory_cache(self, key, value, ttl):
        self.memory_cache[key] = {
            'value': value,
            'expiry': time.time() + ttl
        }
        # 清理过期缓存
        self._clean_memory_cache()
        return True
    
    def _delete_memory_cache(self, key):
        if key in self.memory_cache:
            del self.memory_cache[key]
        return True
    
    def _exists_memory_cache(self, key):
        if key in self.memory_cache:
            if time.time() < self.memory_cache[key]['expiry']:
                return True
            else:
                # 缓存过期
                del self.memory_cache[key]
        return False
    
    def _clean_memory_cache(self):
        """清理过期的内存缓存"""
        now = time.time()
        expired_keys = [key for key, item in self.memory_cache.items() if item['expiry'] < now]
        for key in expired_keys:
            del self.memory_cache[key]
    
    def get_session_cache_key(self, session_id):
        return f"session:{session_id}"
    
    def get_user_cache_key(self, user_id):
        return f"user:{user_id}"
    
    def get_knowledge_cache_key(self, query):
        digest = hashlib.sha256(str(query).encode("utf-8")).hexdigest()
        return f"knowledge:{digest}"
    
    def clear_session_cache(self, session_id):
        key = self.get_session_cache_key(session_id)
        return self.delete(key)
    
    def clear_user_cache(self, user_id):
        key = self.get_user_cache_key(user_id)
        return self.delete(key)

    def clear_knowledge_cache(self):
        """清除所有知识库搜索缓存（上传新文档后调用）"""
        prefix = "knowledge:"
        if self.use_redis:
            try:
                keys = self.redis_client.keys(f"{prefix}*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                print(f"Redis clear knowledge cache error: {e}")
                self._maybe_reconnect()
        # 内存缓存也要清
        now = time.time()
        expired = [k for k in self.memory_cache if k.startswith(prefix)]
        for k in expired:
            del self.memory_cache[k]
