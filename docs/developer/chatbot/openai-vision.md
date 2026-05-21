# File inputs

## Source

- [OpenAI API: File Inputs](https://developers.openai.com/api/docs/guides/file-inputs)

OpenAI models can accept files as `input_file` items. In the Responses API, you can send a file as Base64-encoded data, a file ID returned by the Files API (`/v1/files`), or an external URL.

## How it works

`input_file` processing depends on the file type:

- **PDF files**: On models with vision capabilities, such as `gpt-4o` and later models, the API extracts both text and page images and sends both to the model.
- **Non-PDF document and text files** (for example, `.docx`, `.pptx`, `.txt`, and code files): the API extracts text only.
- **Spreadsheet files** (for example, `.xlsx`, `.csv`, `.tsv`): the API runs a spreadsheet-specific augmentation flow (described below).

Use these related tools when they better match your task:

- Use [File Search](https://developers.openai.com/api/docs/guides/tools-file-search) for retrieval over large files instead of passing them directly as `input_file`.
- Use [Hosted Shell](https://developers.openai.com/api/docs/guides/tools-shell#hosted-shell-quickstart) for spreadsheet-heavy tasks that need detailed analysis, such as aggregations, joins, charting, or custom calculations.

## Non-PDF image and chart limitations

For non-PDF files, the API doesn't extract embedded images or charts into the
model context.

To preserve chart and diagram fidelity, convert the file to PDF first, then
send the PDF as `input_file`.

## How spreadsheet augmentation works

For spreadsheet-like files (such as `.xlsx`, `.xls`, `.csv`, `.tsv`, and
`.iif`), `input_file` uses a spreadsheet-specific augmentation process.

Instead of passing entire sheets to the model, the API parses up to the first
1,000 rows per sheet and adds model-generated summary and header metadata so the
model can work from a smaller, structured view of the data.

## Accepted file types

The following table lists common file types accepted in `input_file`. The full
list of extensions and MIME types appears later on this page.

| Category       | Common extensions                                   |
| -------------- | --------------------------------------------------- |
| PDF files      | `.pdf`                                              |
| Text and code  | `.txt`, `.md`, `.json`, `.html`, `.xml`, code files |
| Rich documents | `.doc`, `.docx`, `.rtf`, `.odt`                     |
| Presentations  | `.ppt`, `.pptx`                                     |
| Spreadsheets   | `.csv`, `.xls`, `.xlsx`                             |

## File URLs



You can provide file inputs by linking external URLs.

Use an external file URL

```bash
curl "https://api.openai.com/v1/responses" \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer $OPENAI_API_KEY" \\
    -d '{
        "model": "gpt-5.5",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Analyze the letter and provide a summary of the key points."
                    },
                    {
                        "type": "input_file",
                        "file_url": "https://www.berkshirehathaway.com/letters/2024ltr.pdf"
                    }
                ]
            }
        ]
    }'
```

```javascript
import OpenAI from "openai";
const client = new OpenAI();

const response = await client.responses.create({
    model: "gpt-5.5",
    input: [
        {
            role: "user",
            content: [
                {
                    type: "input_text",
                    text: "Analyze the letter and provide a summary of the key points.",
                },
                {
                    type: "input_file",
                    file_url: "https://www.berkshirehathaway.com/letters/2024ltr.pdf",
                },
            ],
        },
    ],
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5.5",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Analyze the letter and provide a summary of the key points.",
                },
                {
                    "type": "input_file",
                    "file_url": "https://www.berkshirehathaway.com/letters/2024ltr.pdf",
                },
            ],
        },
    ]
)

print(response.output_text)
```

```csharp
using OpenAI.Files;
using OpenAI.Responses;

string key = Environment.GetEnvironmentVariable("OPENAI_API_KEY")!;
OpenAIResponseClient client = new(model: "gpt-5.5", apiKey: key);

using HttpClient http = new();
using Stream stream = await http.GetStreamAsync("https://www.berkshirehathaway.com/letters/2024ltr.pdf");
OpenAIFileClient files = new(key);
OpenAIFile file = files.UploadFile(stream, "2024ltr.pdf", FileUploadPurpose.UserData);

OpenAIResponse response = (OpenAIResponse)client.CreateResponse([
    ResponseItem.CreateUserMessageItem([
        ResponseContentPart.CreateInputTextPart("Analyze the letter and provide a summary of the key points."),
        ResponseContentPart.CreateInputFilePart(file.Id),
    ]),
]);

Console.WriteLine(response.GetOutputText());
```






## Uploading files

The following example uploads a file with the [Files API](https://developers.openai.com/api/docs/api-reference/files), then references its file ID in a request to the model.



Upload a file

```bash
curl https://api.openai.com/v1/files \\
    -H "Authorization: Bearer $OPENAI_API_KEY" \\
    -F purpose="user_data" \\
    -F file="@draconomicon.pdf"

curl "https://api.openai.com/v1/responses" \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer $OPENAI_API_KEY" \\
    -d '{
        "model": "gpt-5.5",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_id": "file-6F2ksmvXxt4VdoqmHRw6kL"
                    },
                    {
                        "type": "input_text",
                        "text": "What is the first dragon in the book?"
                    }
                ]
            }
        ]
    }'
```

```javascript
import fs from "fs";
import OpenAI from "openai";
const client = new OpenAI();

const file = await client.files.create({
    file: fs.createReadStream("draconomicon.pdf"),
    purpose: "user_data",
});

const response = await client.responses.create({
    model: "gpt-5.5",
    input: [
        {
            role: "user",
            content: [
                {
                    type: "input_file",
                    file_id: file.id,
                },
                {
                    type: "input_text",
                    text: "What is the first dragon in the book?",
                },
            ],
        },
    ],
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

file = client.files.create(
    file=open("draconomicon.pdf", "rb"),
    purpose="user_data"
)

response = client.responses.create(
    model="gpt-5.5",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": file.id,
                },
                {
                    "type": "input_text",
                    "text": "What is the first dragon in the book?",
                },
            ]
        }
    ]
)

print(response.output_text)
```

```csharp
using OpenAI.Files;
using OpenAI.Responses;

string key = Environment.GetEnvironmentVariable("OPENAI_API_KEY")!;
OpenAIResponseClient client = new(model: "gpt-5.5", apiKey: key);

OpenAIFileClient files = new(key);
OpenAIFile file = files.UploadFile("draconomicon.pdf", FileUploadPurpose.UserData);

OpenAIResponse response = (OpenAIResponse)client.CreateResponse([
    ResponseItem.CreateUserMessageItem([
        ResponseContentPart.CreateInputFilePart(file.Id),
        ResponseContentPart.CreateInputTextPart("What is the first dragon in the book?"),
    ]),
]);

Console.WriteLine(response.GetOutputText());
```






## Base64-encoded files

You can also send file inputs as Base64-encoded file data.



Send a Base64-encoded file

```bash
curl "https://api.openai.com/v1/responses" \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer $OPENAI_API_KEY" \\
    -d '{
        "model": "gpt-5",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "draconomicon.pdf",
                        "file_data": "...base64 encoded PDF bytes here..."
                    },
                    {
                        "type": "input_text",
                        "text": "What is the first dragon in the book?"
                    }
                ]
            }
        ]
    }'
```

```javascript
import fs from "fs";
import OpenAI from "openai";
const client = new OpenAI();

const data = fs.readFileSync("draconomicon.pdf");
const base64String = data.toString("base64");

const response = await client.responses.create({
    model: "gpt-5",
    input: [
        {
            role: "user",
            content: [
                {
                    type: "input_file",
                    filename: "draconomicon.pdf",
                    file_data: \`data:application/pdf;base64,\${base64String}\`,
                },
                {
                    type: "input_text",
                    text: "What is the first dragon in the book?",
                },
            ],
        },
    ],
});

console.log(response.output_text);
```

```python
import base64
from openai import OpenAI
client = OpenAI()

with open("draconomicon.pdf", "rb") as f:
    data = f.read()

base64_string = base64.b64encode(data).decode("utf-8")

response = client.responses.create(
    model="gpt-5",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "draconomicon.pdf",
                    "file_data": f"data:application/pdf;base64,{base64_string}",
                },
                {
                    "type": "input_text",
                    "text": "What is the first dragon in the book?",
                },
            ],
        },
    ]
)

print(response.output_text)
```






## Usage considerations

Keep these constraints in mind when you use file inputs:

- **Token usage:** PDF parsing includes both extracted text and page images in context, which can increase token usage. Before deploying at scale, review pricing and token implications. [More on pricing](https://developers.openai.com/api/docs/pricing).
- **File size limits:** A single request can include more than one file, but each file must be under 50 MB. The combined limit across all files in the request is 50 MB.
- **Supported models:** PDF parsing that includes text and page images requires models with vision capabilities, such as `gpt-4o` and later models.
- **File upload purpose:** You can upload files with any supported [purpose](https://developers.openai.com/api/docs/api-reference/files/create#files-create-purpose), but use `user_data` for files you plan to pass as model inputs.

## Full list of accepted file types

| Category       | Extensions                                                                                                                                                                                                                                                                                                                                                 | MIME types                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PDF files      | PDF files (`.pdf`)                                                                                                                                                                                                                                                                                                                                         | `application/pdf`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Spreadsheets   | Excel sheets (`.xla`, `.xlb`, `.xlc`, `.xlm`, `.xls`, `.xlsx`, `.xlt`, `.xlw`)                                                                                                                                                                                                                                                                             | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`, `application/vnd.ms-excel`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Spreadsheets   | CSV / TSV / IIF (`.csv`, `.tsv`, `.iif`), Google Sheets                                                                                                                                                                                                                                                                                                    | `text/csv`, `application/csv`, `text/tsv`, `text/x-iif`, `application/x-iif`, `application/vnd.google-apps.spreadsheet`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Rich documents | Word/ODT/RTF docs (`.doc`, `.docx`, `.dot`, `.odt`, `.rtf`), Pages, Google Docs                                                                                                                                                                                                                                                                            | `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `application/msword`, `application/rtf`, `text/rtf`, `application/vnd.oasis.opendocument.text`, `application/vnd.apple.pages`, `application/vnd.google-apps.document`, `application/vnd.apple.iwork`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Presentations  | PowerPoint slides (`.pot`, `.ppa`, `.pps`, `.ppt`, `.pptx`, `.pwz`, `.wiz`), Keynote, Google Slides                                                                                                                                                                                                                                                        | `application/vnd.openxmlformats-officedocument.presentationml.presentation`, `application/vnd.ms-powerpoint`, `application/vnd.apple.keynote`, `application/vnd.google-apps.presentation`, `application/vnd.apple.iwork`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Text and code  | Text/code formats (`.asm`, `.bat`, `.c`, `.cc`, `.conf`, `.cpp`, `.css`, `.cxx`, `.def`, `.dic`, `.eml`, `.h`, `.hh`, `.htm`, `.html`, `.ics`, `.ifb`, `.in`, `.js`, `.json`, `.ksh`, `.list`, `.log`, `.markdown`, `.md`, `.mht`, `.mhtml`, `.mime`, `.mjs`, `.nws`, `.pl`, `.py`, `.rst`, `.s`, `.sql`, `.srt`, `.text`, `.txt`, `.vcf`, `.vtt`, `.xml`) | `application/javascript`, `application/typescript`, `text/xml`, `text/x-shellscript`, `text/x-rst`, `text/x-makefile`, `text/x-lisp`, `text/x-asm`, `text/vbscript`, `text/css`, `message/rfc822`, `application/x-sql`, `application/x-scala`, `application/x-rust`, `application/x-powershell`, `text/x-diff`, `text/x-patch`, `application/x-patch`, `text/plain`, `text/markdown`, `text/x-java`, `text/x-script.python`, `text/x-python`, `text/x-c`, `text/x-c++`, `text/x-golang`, `text/html`, `text/x-php`, `application/x-php`, `application/x-httpd-php`, `application/x-httpd-php-source`, `text/x-ruby`, `text/x-sh`, `text/x-bash`, `application/x-bash`, `text/x-zsh`, `text/x-tex`, `text/x-csharp`, `application/json`, `text/x-typescript`, `text/javascript`, `text/x-go`, `text/x-rust`, `text/x-scala`, `text/x-kotlin`, `text/x-swift`, `text/x-lua`, `text/x-r`, `text/x-R`, `text/x-julia`, `text/x-perl`, `text/x-objectivec`, `text/x-objectivec++`, `text/x-erlang`, `text/x-elixir`, `text/x-haskell`, `text/x-clojure`, `text/x-groovy`, `text/x-dart`, `text/x-awk`, `application/x-awk`, `text/jsx`, `text/tsx`, `text/x-handlebars`, `text/x-mustache`, `text/x-ejs`, `text/x-jinja2`, `text/x-liquid`, `text/x-erb`, `text/x-twig`, `text/x-pug`, `text/x-jade`, `text/x-tmpl`, `text/x-cmake`, `text/x-dockerfile`, `text/x-gradle`, `text/x-ini`, `text/x-properties`, `text/x-protobuf`, `application/x-protobuf`, `text/x-sql`, `text/x-sass`, `text/x-scss`, `text/x-less`, `text/x-hcl`, `text/x-terraform`, `application/x-terraform`, `text/x-toml`, `application/x-toml`, `application/graphql`, `application/x-graphql`, `text/x-graphql`, `application/x-ndjson`, `application/json5`, `application/x-json5`, `text/x-yaml`, `application/toml`, `application/x-yaml`, `application/yaml`, `text/x-astro`, `text/srt`, `application/x-subrip`, `text/x-subrip`, `text/vtt`, `text/x-vcard`, `text/calendar` |

## Next steps

Next, you might want to explore one of these resources:

<div>
  [

<span slot="icon">
        </span>
      Use the Playground to develop and iterate on prompts with file inputs.

](https://platform.openai.com/chat/edit)
</div>

<div>
  [

<span slot="icon">
        </span>
      Check out the API reference for more options.

](https://developers.openai.com/api/docs/api-reference/responses)
</div>

<div>
  [

<span slot="icon">
        </span>
      Use retrieval over chunked files when you need scalable search instead of
      sending whole files in a single context window.

](https://developers.openai.com/api/docs/guides/tools-file-search)
</div>

<div>
  [

<span slot="icon">
        </span>
      Use Hosted Shell for advanced spreadsheet workflows such as joins,
      aggregations, and charting.

](https://developers.openai.com/api/docs/guides/tools-shell#hosted-shell-quickstart)
</div>
