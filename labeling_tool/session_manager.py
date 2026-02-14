import requests
import time
import tomli
import tomli_w
import os
from datetime import datetime


class SessionManager:
    def __init__(self, config_path="config.toml"):
        self.config_path = config_path
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        self.config = self._load_config(config_path)
        self.visited_login = set()

    def _load_config(self, path):
        if not os.path.exists(path):
            if os.path.exists("config.example.toml"):
                print(f"Config {path} not found, using config.example.toml")
                path = "config.example.toml"
            else:
                # If neither exists, start empty (though this might break things)
                return {}

        with open(path, "rb") as f:
            return tomli.load(f)

    def save_config(self):
        try:
            # Always save to the target path (e.g. config.toml), even if we loaded from example
            with open(self.config_path, "wb") as f:
                tomli_w.dump(self.config, f)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def get_sources(self):
        """Returns a dict of label -> key"""
        sources = {}
        if "sources" in self.config:
            for key, conf in self.config["sources"].items():
                label = conf.get("label", key)
                sources[label] = key
        return sources

    def get_last_source(self):
        return self.config.get("last_source")

    def set_last_source(self, source_label):
        self.config["last_source"] = source_label
        self.save_config()

    def ensure_login(self, source_key):
        conf = self.config["sources"].get(source_key)
        if not conf:
            return

        login_url = conf.get("login_url")
        if login_url and source_key not in self.visited_login:
            print(f"Initializing session for {source_key}: Visiting {login_url}...")
            try:
                self.session.get(login_url, timeout=10)
                self.visited_login.add(source_key)
                print("Session initialized.")
            except Exception as e:
                print(f"Failed to initialize session: {e}")
                raise

    def fetch_captcha(self, source_key):
        self.ensure_login(source_key)

        conf = self.config["sources"].get(source_key)
        if not conf:
            raise ValueError(f"Unknown source: {source_key}")

        url_template = conf["captcha_url"]

        # Generate Timestamps
        now = datetime.now()
        ts_js = now.strftime("%a %b %d %Y %H:%M:%S GMT+0800 (China Standard Time)")
        ts_ms = int(time.time() * 1000)

        url = url_template.replace("{timestamp_js}", ts_js).replace(
            "{timestamp_ms}", str(ts_ms)
        )

        return self.session.get(url, timeout=10)

    def get_save_dir(self, source_key):
        """Returns preferred save directory for source, defaults to 'raw_captchas'"""
        conf = self.config["sources"].get(source_key)
        if conf and "save_dir" in conf:
            return conf["save_dir"]
        return "raw_captchas"
